import json

from agro_phenology.external_validation_collection import check, _validate_source
from agro_phenology.external_validation_io import file_inventory, write_json
import pytest


def test_manifest_detects_changed_observations(tmp_path):
    (tmp_path / "raw.tsv").write_text("date\tvalue\n2020-06-01\t0\n")
    write_json(tmp_path / "manifest.json", {"status": "complete_with_documented_gaps", "file_inventory": file_inventory(tmp_path)})
    assert check(tmp_path)["status"] == "passed"
    (tmp_path / "raw.tsv").write_text("date\tvalue\n2020-06-01\t1\n")
    assert check(tmp_path)["status"] == "failed"


def test_source_files_cannot_escape_or_clobber_each_other():
    with pytest.raises(ValueError):
        _validate_source({"source_id": "source", "files": [{"name": "../../secret"}]})
    with pytest.raises(ValueError):
        _validate_source({"source_id": "source", "files": [{"name": "a"}, {"name": "a"}]})
