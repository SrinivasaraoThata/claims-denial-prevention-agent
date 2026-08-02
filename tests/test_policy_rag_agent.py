import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.policy_rag_agent import build_index, check_policy, chunk_policy_docs  # noqa: E402

TEST_PERSIST_DIR = Path(__file__).resolve().parent / "_tmp_chroma_db"


@pytest.fixture(scope="module")
def collection():
    shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)
    collection = build_index(persist_dir=TEST_PERSIST_DIR)
    yield collection
    shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)


def test_chunk_policy_docs_covers_every_file():
    chunks = chunk_policy_docs()
    docs_dir = Path(__file__).resolve().parents[1] / "data" / "policy_docs"
    assert len(chunks) == len(list(docs_dir.glob("*.md")))
    assert all(chunk.text for chunk in chunks)


def test_retrieval_surfaces_matching_policy_doc(collection):
    result = check_policy({"procedure_code": "70551", "member_plan_id": "PLAN001"}, collection)
    assert "Prior Authorization: Advanced Imaging" in result.sources


def test_retrieval_for_surgery_code(collection):
    result = check_policy({"procedure_code": "29881", "member_plan_id": "PLAN004"}, collection)
    assert "Prior Authorization: Surgical Procedures" in result.sources


def test_no_api_key_falls_back_to_extractive_answer(collection, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = check_policy({"procedure_code": "45378", "member_plan_id": "PLAN002"}, collection)
    assert not result.llm_used
    assert result.answer
    assert result.chunks
