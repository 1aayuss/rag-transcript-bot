import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from qdrant_client.models import PointStruct
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding, SparseTextEmbedding

load_dotenv()


def load_all_pdfs(pdfs: List[str]) -> List[Dict[str, Any]]:
    docs = []
    for path in pdfs:
        if not os.path.exists(path):
            print(f"[warn] file not found: {path}")
            continue
        for d in PyPDFium2Loader(path).load():
            d.metadata = {"source": os.path.basename(path)}
            docs.append(d)
    if not docs:
        raise FileNotFoundError("No valid PDFs were loaded.")
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
    return splitter.split_documents(docs)

# CREATE COLLECTION


def create_collection(client: QdrantClient, name: str, dense_size: int = 768):
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(
            size=dense_size, distance=Distance.COSINE, on_disk=True)},
        sparse_vectors_config={"sparse": SparseVectorParams(
            modifier=models.Modifier.IDF)},
        on_disk_payload=True,
    )
    client.create_payload_index(
        collection_name=name, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD)
    print(f"[ok] created hybrid collection '{name}'")


def main():
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_API_KEY")
    collection_name = os.environ.get("QDRANT_COLLECTION")
    batch_size = 50

    client = QdrantClient(url=url, api_key=key, timeout=300)

    pdfs = [
        "./data/1sep.pdf",
        "./data/2sep.pdf",
        "./data/3sep.pdf",
        "./data/4sep.pdf",
        "./data/8sep.pdf",
        "./data/9sep.pdf",
        "./data/11sep.pdf"
    ]

    print("loading PDFs…")
    docs = load_all_pdfs(pdfs)
    print(f"Total pages: {len(docs)}")

    print("splitting…")
    chunks = split_docs(docs)
    print(f"Total chunks: {len(chunks)}")

    print("creating collection…")
    create_collection(client, collection_name, dense_size=768)

    print("generating embeddings (dense + sparse)")
    dense_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
    sparse_model = SparseTextEmbedding("Qdrant/BM25")

    dense_vecs = list(dense_model.embed(d.page_content for d in chunks))
    sparse_vecs = list(sparse_model.embed(d.page_content for d in chunks))

    print("upserting")
    points = []
    for idx in range(len(chunks)):
        point = PointStruct(
            id=idx,
            vector={
                "dense": dense_vecs[idx],
                "sparse": sparse_vecs[idx].as_object()
            },
            payload={
                "text": chunks[idx].page_content,
                "source": str(chunks[idx].metadata.get("source", "unknown"))
            }
        )
        points.append(point)

    total_upserted = 0

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True
            )
            total_upserted += len(batch)
            print(
                f"[progress] upserted batch {i//batch_size + 1}: {len(batch)} points (total: {total_upserted}/{len(points)})")
        except Exception as e:
            print(f"[error] failed to upsert batch {i//batch_size + 1}: {e}")
            raise

    print(
        f"successfully upserted all {total_upserted} points into '{collection_name}'")


if __name__ == "__main__":
    main()
