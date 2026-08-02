"""Policy RAG agent: retrieves relevant policy text for a claim and grounds
an answer in it (e.g. "does this procedure need prior authorization?").

Retrieval uses Chroma with a TF-IDF embedding function fit on the policy
corpus. TF-IDF is offline and deterministic (no model download, no network
call), which keeps indexing and tests fast and reproducible; it's a
reasonable fit for a small, fixed corpus of policy documents that are all
in-domain insurance language. Answer synthesis uses Gemini when a
GOOGLE_API_KEY is configured, grounded strictly in the retrieved chunks. If
no key is configured, the agent falls back to returning the retrieved
chunks directly rather than failing, so the rest of the pipeline still
works without a paid/keyed dependency.
"""

import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

POLICY_DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "policy_docs"
DEFAULT_PERSIST_DIR = Path(
    os.environ.get("CHROMA_PERSIST_DIR", Path(__file__).resolve().parents[1] / "data" / "chroma_db")
)
COLLECTION_NAME = "policy_docs"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"

TITLE_RE = re.compile(r"^# +(.+)$", re.MULTILINE)


@dataclass
class PolicyChunk:
    chunk_id: str
    source: str
    text: str


@dataclass
class PolicyRagResult:
    query: str
    answer: str
    llm_used: bool
    sources: list[str] = field(default_factory=list)
    chunks: list[PolicyChunk] = field(default_factory=list)


def chunk_policy_docs(docs_dir: Path = POLICY_DOCS_DIR) -> list[PolicyChunk]:
    """Load each policy markdown file as a single chunk.

    Whole documents (each ~300-400 words, one per procedure category) are
    used as the retrieval unit rather than per-section splits. Splitting by
    section separates the "which procedure codes this applies to" scope
    section from the "what's required by plan" section, which breaks
    retrieval for a query built from a procedure code: nothing in the
    correct chunk would mention that code. Docs here are short enough that
    whole-document retrieval doesn't lose useful precision.
    """
    chunks = []
    for doc_path in sorted(docs_dir.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        title_match = TITLE_RE.search(text)
        title = title_match.group(1).strip() if title_match else doc_path.stem
        chunks.append(PolicyChunk(chunk_id=doc_path.stem, source=title, text=text.strip()))
    return chunks


class TfidfEmbeddingFunction:
    """Chroma-compatible embedding function backed by a fitted TfidfVectorizer."""

    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer

    def __call__(self, input):  # noqa: A002 - chromadb's EmbeddingFunction interface
        return self.vectorizer.transform(input).toarray().tolist()

    def name(self) -> str:
        return "tfidf"


def build_index(
    persist_dir: Path = DEFAULT_PERSIST_DIR, docs_dir: Path = POLICY_DOCS_DIR
) -> chromadb.api.models.Collection.Collection:
    """(Re)build the Chroma collection for the policy corpus."""
    chunks = chunk_policy_docs(docs_dir)
    corpus = [chunk.text for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(corpus)

    persist_dir.mkdir(parents=True, exist_ok=True)
    with open(persist_dir / VECTORIZER_FILENAME, "wb") as f:
        pickle.dump(vectorizer, f)

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass

    embedding_fn = TfidfEmbeddingFunction(vectorizer)
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[{"source": chunk.source} for chunk in chunks],
    )
    return collection


def load_collection(persist_dir: Path = DEFAULT_PERSIST_DIR):
    """Load a previously built collection, or build one if none exists."""
    vectorizer_path = persist_dir / VECTORIZER_FILENAME
    if not vectorizer_path.exists():
        return build_index(persist_dir)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    client = chromadb.PersistentClient(path=str(persist_dir))
    embedding_fn = TfidfEmbeddingFunction(vectorizer)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def _build_query(claim: dict) -> str:
    # Kept terse and keyword-heavy rather than a natural-language sentence:
    # generic phrasing ("coverage requirements for procedure code X under
    # plan Y") dilutes TF-IDF similarity with words common to every policy
    # doc, and the procedure code / plan id are what actually distinguish
    # the right document.
    return f"{claim.get('procedure_code')} {claim.get('member_plan_id')} prior authorization requirement"


def _extractive_answer(chunks: list[PolicyChunk]) -> str:
    if not chunks:
        return "No relevant policy text found."
    return "\n\n".join(chunk.text for chunk in chunks)


def _gemini_answer(query: str, chunks: list[PolicyChunk]) -> str | None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    context = "\n\n".join(chunk.text for chunk in chunks)
    prompt = (
        "Answer the question using only the policy excerpts below. If the "
        "excerpts don't cover it, say so explicitly. Be concise.\n\n"
        f"Policy excerpts:\n{context}\n\nQuestion: {query}"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def check_policy(claim: dict, collection, k: int = 2) -> PolicyRagResult:
    query = _build_query(claim)

    results = collection.query(query_texts=[query], n_results=k)
    chunks = [
        PolicyChunk(
            chunk_id=results["ids"][0][i],
            source=results["metadatas"][0][i]["source"],
            text=results["documents"][0][i],
        )
        for i in range(len(results["ids"][0]))
    ]

    llm_answer = _gemini_answer(query, chunks)
    if llm_answer is not None:
        return PolicyRagResult(
            query=query,
            answer=llm_answer,
            llm_used=True,
            sources=[chunk.source for chunk in chunks],
            chunks=chunks,
        )

    return PolicyRagResult(
        query=query,
        answer=_extractive_answer(chunks),
        llm_used=False,
        sources=[chunk.source for chunk in chunks],
        chunks=chunks,
    )
