import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, QuantizationConfig, BinaryQuantization
from qdrant_client.models import PointStruct
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from fastembed import TextEmbedding, SparseTextEmbedding

load_dotenv()


def load_all_pdfs(pdfs: List[str], user: str = "system") -> List[Document]:
    """Load PDFs and create Documents with enhanced metadata."""
    docs = []
    current_timestamp = datetime.now().isoformat()
    current_date = datetime.now().strftime("%Y-%m-%d")

    for path in pdfs:
        if not os.path.exists(path):
            print(f"[warn] file not found: {path}")
            continue
        for d in PyPDFium2Loader(path).load():
            # Enhanced metadata with Date, User, Source, Timestamp
            d.metadata = {
                "source": os.path.basename(path),
                "user": user,
                "date": current_date,
                "timestamp": current_timestamp,
                "page": d.metadata.get("page", 0)
            }
            docs.append(d)
    if not docs:
        raise FileNotFoundError("No valid PDFs were loaded.")
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
    return splitter.split_documents(docs)

# CREATE COLLECTION WITH BINARY QUANTIZATION


def create_collection(client: QdrantClient, name: str, dense_size: int = 768):
    """Create collection with binary quantization for efficient indexing."""
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(
                size=dense_size,
                distance=Distance.COSINE,
                on_disk=True,
                quantization_config=BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                )
            )
        },
        sparse_vectors_config={"sparse": SparseVectorParams(
            modifier=models.Modifier.IDF)},
        on_disk_payload=True,
    )

    client.create_payload_index(
        collection_name=name,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=name,
        field_name="timestamp",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=name,
        field_name="user",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=name,
        field_name="date",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    print(f"[ok] created hybrid collection '{name}' with binary quantization")


def main():
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_API_KEY")
    collection_name = os.environ.get("QDRANT_COLLECTION")
    batch_size = 50

    client = QdrantClient(url=url, api_key=key, timeout=300)

    data_dir = "../data"
    pdfs = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    print("loading PDFs…")
    docs = load_all_pdfs(pdfs)
    print(f"Total pages: {len(docs)}")

    print("splitting…")
    chunks = split_docs(docs)
    print(f"Total chunks: {len(chunks)}")

    print("creating collection…")
    create_collection(client, collection_name, dense_size=768)

    print("initializing embedding models")
    dense_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
    sparse_model = SparseTextEmbedding("Qdrant/BM25")

    print(
        f"processing {len(chunks)} chunks in batches of {batch_size}")
    total_uploaded = 0

    for batch_idx in range(0, len(chunks), batch_size):
        batch_end = min(batch_idx + batch_size, len(chunks))
        batch_chunks = chunks[batch_idx:batch_end]

        print(
            f"[batch {batch_idx//batch_size + 1}] generating embeddings for chunks {batch_idx}-{batch_end-1}")

        batch_texts = [chunk.page_content for chunk in batch_chunks]
        dense_vecs = list(dense_model.embed(batch_texts))
        sparse_vecs = list(sparse_model.embed(batch_texts))

        points = []
        for i, chunk in enumerate(batch_chunks):
            metadata = chunk.metadata
            point = PointStruct(
                id=batch_idx + i,
                vector={
                    "dense": dense_vecs[i],
                    "sparse": sparse_vecs[i].as_object()
                },
                payload={
                    "text": chunk.page_content,
                    "source": str(metadata.get("source", "unknown")),
                    "user": str(metadata.get("user", "system")),
                    "date": str(metadata.get("date", "")),
                    "timestamp": str(metadata.get("timestamp", "")),
                    "page": metadata.get("page", 0)
                }
                # prompt + metadata to the payload
            )
            points.append(point)

        try:
            client.upload_points(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            total_uploaded += len(points)
            print(
                f"[batch {batch_idx//batch_size + 1}] uploaded {len(points)} points (total: {total_uploaded}/{len(chunks)})")
        except Exception as e:
            print(
                f"[error] failed to upload batch {batch_idx//batch_size + 1}: {e}")
            raise

        del dense_vecs, sparse_vecs, points, batch_texts

    print(
        f"\n✅ successfully uploaded all {total_uploaded} points into '{collection_name}'")


if __name__ == "__main__":
    main()
