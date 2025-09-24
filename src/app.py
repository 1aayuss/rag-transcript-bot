import os
from opik.integrations.langchain import OpikTracer
from dotenv import load_dotenv
import streamlit as st
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from schemas import RAGAnswer, Context

load_dotenv()


COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "transcripts-v2")
TOP_K = 4
MMR_DIVERSITY = 0.3
DENSE_MODEL = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
SPARSE_MODEL = SparseTextEmbedding("Qdrant/BM25")

# RETRIEVAL


def retrieve_ctx(
    client: QdrantClient,
    question: str,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dense_q = next(DENSE_MODEL.query_embed(question))
    sparse_q = next(SPARSE_MODEL.query_embed(question))

    prefetch = [
        models.Prefetch(query=dense_q, using="dense",
                        limit=max(10, TOP_K * 3)),
        models.Prefetch(
            query=models.SparseVector(**sparse_q.as_object()),
            using="sparse",
            limit=max(10, TOP_K * 3),
        ),
    ]

    qfilter = None
    if source_filter:
        qfilter = models.Filter(
            must=[models.FieldCondition(
                key="source", match=models.MatchValue(value=source_filter))]
        )

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.NearestQuery(
            nearest=dense_q,
            mmr=models.Mmr(diversity=MMR_DIVERSITY, candidates_limit=50),
        ),
        using="dense",
        with_payload=True,
        limit=TOP_K,
        query_filter=qfilter,
    )

    payloads: List[Dict[str, Any]] = []
    for p in res.points:
        payload = dict(p.payload or {})
        payloads.append(payload)
    return payloads


# GENERATION

def generate_res(
    llm: ChatGoogleGenerativeAI,
    question: str,
    contexts: List[Dict[str, Any]],
) -> RAGAnswer:
    system_prompt = """
        You are an expert assistant who only answers questions related to Artificial Intelligence and Machine Learning.
        For every question, you must first check if the answer can be found in the provided context.
        If the answer is present, generate a helpful and accurate response by quoting the relevant speaker's statement(s) from the context in a clear and improved way.
        If the question is not covered in the context, simply reply: "I don't know, we didn't discuss this."
    """
    user_prompt = """
        Carefully review the CONTEXT below and the USER's QUESTION.
        If the context contains information that answers the question, respond by quoting the speaker's statement(s) that best address the question, and present them in a clear and improved way.
        If the context does not contain an answer, reply with: "I don't know, we didn't discuss this."
        <context>
        CONTEXT: {context}
        </context>

        <question>
        QUESTION: {question}
        </question>
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt)])

    snippets: List[str] = []
    ctx_items: List[Context] = []
    for c in contexts:
        txt = str(c.get("text", "")).strip()
        src = str(c.get("source", "")).strip()
        if txt:
            snippets.append(txt)
            ctx_items.append(Context(text=txt[:400], source=src))

    structured_llm = llm.with_structured_output(RAGAnswer)
    rag: RAGAnswer = structured_llm.invoke(
        prompt.format_messages(
            context="\n\n---\n\n".join(snippets), question=question)
    )
    rag.contexts = ctx_items
    return rag


def main():
    qurl = os.environ.get("QDRANT_URL")
    qkey = os.environ.get("QDRANT_API_KEY")

    client = QdrantClient(url=qurl, api_key=qkey)

    tracer = OpikTracer()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.0,
        max_tokens=None,
        callbacks=[tracer]
    )

    st.set_page_config(page_title="Python and AI Power-up", layout="centered")
    st.title("Python and AI Power-up Program")

    with st.sidebar:
        st.header("Settings")
        st.caption(f"Collection: `{COLLECTION_NAME}`")
        data_files = [
            "All",
            "1sep.pdf",
            "2sep.pdf",
            "3sep.pdf",
            "4sep.pdf",
            "8sep.pdf",
            "9sep.pdf",
            "11sep.pdf"
        ]
        source_filter = st.selectbox("Filter by filename", data_files)
        if source_filter == "All":
            source_filter = ""

    with st.form("question_form"):
        question = st.text_input("Ask a question")
        ask = st.form_submit_button("Generate")

    print(question)
    print(question.strip())
    if ask and question.strip():
        with st.spinner("Retrieving…"):
            ctx = retrieve_ctx(
                client=client,
                question=question.strip(),
                source_filter=source_filter.strip() or None,
            )

        with st.spinner("Generating answer…"):
            rag = generate_res(
                llm=llm, question=question.strip(), contexts=ctx)

        st.subheader("Answer")
        st.write(rag.answer or "I don't know.")

        st.subheader("Contexts")
        for i, c in enumerate(rag.contexts, 1):
            st.markdown(f"**[{i}]** `{c.source}`")
            st.caption(c.text)


if __name__ == "__main__":
    main()
