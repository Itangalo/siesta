import pytest

from backend.siesta import api as api_module


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Keep tests out of the real game data: sessions and feedback written by
    the API go to a per-test temporary directory."""
    data_dir = tmp_path / "api-data"  # tests may use tmp_path themselves
    monkeypatch.setattr(api_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(api_module, "FEEDBACK_LOG_PATH", data_dir / "human_feedback.jsonl")
    monkeypatch.setattr(api_module, "store", api_module.SessionStore(storage_dir=data_dir / "sessions"))
