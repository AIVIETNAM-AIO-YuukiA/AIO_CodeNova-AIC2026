from pathlib import Path
from unittest.mock import patch

from indexing.manifest import JsonlManifest


def test_replace_all_writes_rows(tmp_path: Path) -> None:
    manifest = JsonlManifest(tmp_path / "test.jsonl")
    manifest.replace_all([{"id": "1"}, {"id": "2"}], unique_key="id")

    assert manifest.read_all() == [{"id": "1"}, {"id": "2"}]


def test_replace_all_retries_transient_permission_error(tmp_path: Path) -> None:
    # Windows can transiently raise PermissionError/WinError 5 on os.replace()
    # when another process (antivirus, search indexer) briefly holds the file.
    manifest = JsonlManifest(tmp_path / "test.jsonl")

    original_replace = Path.replace
    call_count = 0

    def flaky_replace(self: Path, target: Path) -> Path:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise PermissionError("simulated transient lock")
        return original_replace(self, target)

    with patch.object(Path, "replace", flaky_replace):
        manifest.replace_all([{"id": "1"}], unique_key="id")

    assert call_count == 3
    assert manifest.read_all() == [{"id": "1"}]


def test_replace_all_reraises_after_exhausting_retries(tmp_path: Path) -> None:
    manifest = JsonlManifest(tmp_path / "test.jsonl")

    def always_fails(self: Path, target: Path) -> Path:
        raise PermissionError("persistent lock")

    with patch.object(Path, "replace", always_fails):
        try:
            manifest.replace_all([{"id": "1"}], unique_key="id")
            raise AssertionError("expected PermissionError to propagate")
        except PermissionError:
            pass

    # No partial/corrupt file left behind after the retries give up.
    assert not manifest.path.exists()
