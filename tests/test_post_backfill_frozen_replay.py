from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agro_phenology.early_warning_core import sha256_file
from agro_phenology.post_backfill_frozen_replay import (
    C6_POLICY_FAMILIES,
    DEFAULT_CYCLE3,
    EXPECTED_CYCLE3_MANIFEST_SHA256,
    RUN_FORMAT,
    _c6_runtime,
    _ensure_output_directory,
    _reproduce_text,
    output_directory,
    pair_scenarios,
    read_c6_policy_records,
    replay_identity_audit,
    training_population_audit,
    verify_frozen_run,
)


def test_reproduce_command_contains_no_patch_markers(tmp_path: Path) -> None:
    text = _reproduce_text(
        tmp_path / "run",
        v3_dir=tmp_path / "v3",
        v4_dir=tmp_path / "v4",
        cycle3_dir=tmp_path / "cycle3",
        frozen_era_path=tmp_path / "frozen.parquet",
        extended_era_path=tmp_path / "extended.parquet",
        nasa_daily_path=tmp_path / "nasa.parquet",
        nasa_mapping_path=tmp_path / "mapping.parquet",
    )

    assert "\n+  --" not in text
    assert 'new_run_dir="results/late_blight_early_warning/' in text
    assert '--run-dir "$new_run_dir"' in text
    assert "\\\n  --v3-dir" in text


def _write_manifest(run_dir: Path, hash_style: str = "mapping") -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "artifact.txt"
    artifact.write_text("frozen\n", encoding="utf-8")
    digest = sha256_file(artifact)
    metadata: object
    if hash_style == "mapping":
        metadata = {"sha256": digest, "bytes": artifact.stat().st_size}
    else:
        metadata = digest
    manifest = {
        "format": "fixture",
        "status": "complete",
        "output_hashes": {"artifact.txt": metadata},
    }
    path = run_dir / "execution_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def test_output_is_always_nested_and_unknown_nonempty_directory_is_refused(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "combined_run"
    assert output_directory(parent) == parent / "frozen_replay"

    unknown = output_directory(parent)
    unknown.mkdir(parents=True)
    (unknown / "someone-elses-file.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="unrecognised"):
        _ensure_output_directory(parent)

    clean_parent = tmp_path / "clean"
    target = _ensure_output_directory(clean_parent)
    marker = json.loads((target / ".frozen_replay_run.json").read_text())
    assert target == clean_parent / "frozen_replay"
    assert marker["format"] == RUN_FORMAT


@pytest.mark.parametrize("hash_style", ["mapping", "string"])
def test_frozen_verifier_accepts_both_parent_hash_schemas(
    tmp_path: Path, hash_style: str
) -> None:
    run_dir = tmp_path / hash_style
    manifest_hash = _write_manifest(run_dir, hash_style)
    audit = verify_frozen_run(
        run_dir, manifest_hash, expected_format="fixture"
    )
    assert audit["status"] == "passed"
    assert audit["outputs_checked"] == 1

    (run_dir / "artifact.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="output hash failures"):
        verify_frozen_run(run_dir, manifest_hash)


def _policy_row(family: str, threshold: float, *, enabled: bool) -> dict[str, object]:
    return {
        "fold_id": "test_2020",
        "score_model": "C6_weather",
        "evaluation_scope": "service_calendar",
        "candidate_family": family,
        "alpha": 0.0,
        "threshold": threshold,
        "active_days": 7,
        "cooldown_days": 15,
        "minimum_repeat_interval_days": 7,
        "growth_override_enabled": enabled,
        "growth_logit_delta": np.log(1.5) if enabled else np.nan,
    }


def test_c6_policy_reader_uses_exact_allow_list_without_ranking(tmp_path: Path) -> None:
    rows = [
        _policy_row("P0_saved", 0.10, enabled=False),
        _policy_row("P_growth_selected", 0.20, enabled=True),
        # A distractor that would look attractive to a selector must be ignored.
        {
            **_policy_row("P0_selected", 0.99, enabled=False),
            "timely_recall": 1.0,
        },
    ]
    path = tmp_path / "policy_selection.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    records = read_c6_policy_records(path, years=[2020])

    assert set(records) == {
        ("test_2020", "P0_saved"),
        ("test_2020", "P_growth_selected"),
    }
    assert records[("test_2020", "P0_saved")]["threshold"] == pytest.approx(0.1)
    assert records[("test_2020", "P_growth_selected")]["threshold"] == pytest.approx(
        0.2
    )


def test_default_cycle3_contains_both_frozen_c6_families_for_every_main_fold() -> None:
    assert sha256_file(DEFAULT_CYCLE3 / "execution_manifest.json") == (
        EXPECTED_CYCLE3_MANIFEST_SHA256
    )
    records = read_c6_policy_records(DEFAULT_CYCLE3 / "policy_selection.csv")
    assert len(records) == 12
    for year in range(2020, 2026):
        for family in C6_POLICY_FAMILIES:
            assert (f"test_{year}", family) in records
    assert records[("test_2020", "P0_saved")]["threshold"] == pytest.approx(0.1)
    assert records[("test_2025", "P_growth_selected")]["threshold"] == pytest.approx(
        0.2
    )


class _FakeBundle:
    def __init__(self, alpha: float, available: list[bool]):
        self.alpha = alpha
        self.available = np.asarray(available, dtype=bool)

    def predict(self, frame: pd.DataFrame) -> SimpleNamespace:
        assert len(frame) == len(self.available)
        return SimpleNamespace(
            actionable_probability=np.asarray([0.2, 0.3, 0.4]),
            correction_available=self.available,
        )


def test_c6_runtime_preserves_frozen_alpha_zero_and_fallback_semantics() -> None:
    frame = pd.DataFrame({"service_active": [True, True, False]})
    alpha_zero = _c6_runtime(frame, _FakeBundle(0.0, [False, True, False]))
    assert alpha_zero["score"].tolist()[:2] == pytest.approx([0.2, 0.3])
    assert pd.isna(alpha_zero["score"].iloc[2])
    assert alpha_zero["origin"].tolist()[:2] == ["C0", "C0"]
    assert not alpha_zero["fallback"].any()

    corrected = _c6_runtime(frame, _FakeBundle(1.0, [False, True, False]))
    assert corrected["origin"].tolist()[:2] == ["C0_fallback", "C6_weather"]
    assert corrected["fallback"].tolist() == [True, False, True]


def test_pairing_requires_same_population_and_records_transitions() -> None:
    old = pd.DataFrame(
        {
            "key": [1, 2],
            "score": [np.nan, 0.5],
            "correction_available": [False, True],
            "message_issued": [False, True],
            "alarm_active": [False, True],
        }
    )
    new = pd.DataFrame(
        {
            "key": [1, 2],
            "score": [0.7, 0.4],
            "correction_available": [True, True],
            "message_issued": [True, False],
            "alarm_active": [True, True],
        }
    )
    columns = [
        "score",
        "correction_available",
        "message_issued",
        "alarm_active",
    ]
    paired = pair_scenarios(old, new, keys=["key"], value_columns=columns)
    assert paired["score_became_computable"].tolist() == [True, False]
    assert paired["availability_transition"].tolist() == [
        "gained_available",
        "stable_available",
    ]
    assert paired["message_transition"].tolist() == [
        "gained_message",
        "lost_message",
    ]
    assert paired["alarm_transition"].tolist() == [
        "gained_alarm",
        "stable_alarm",
    ]

    with pytest.raises(AssertionError, match="populations differ"):
        pair_scenarios(old, new.iloc[:1], keys=["key"], value_columns=columns)


def _audit_frame(*, complete_second: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "field_season": ["known", "unknown"],
            "season": [2020, 2020],
            "issue_date": pd.to_datetime(["2020-07-01", "2020-07-02"]),
            "target_observable": [True, False],
            "service_active": [True, True],
            "candidate_comparison_complete": [True, complete_second],
            "target_class": ["actionable", "unknown"],
        }
    )


def test_training_audit_does_not_count_weather_only_unknown_tail_as_supervised() -> None:
    contract = {
        "rolling_origin_folds": [
            {
                "id": "test_2020",
                "train_years": [2020, 2020],
                "validation_years": [2020, 2020],
                "test_years": [2020, 2020],
            }
        ]
    }
    audit = training_population_audit(
        _audit_frame(complete_second=False),
        _audit_frame(complete_second=True),
        contract,
        years=[2020],
    )
    assert audit["eligible_rows_old"].tolist() == [1, 1, 1]
    assert audit["eligible_rows_new"].tolist() == [1, 1, 1]
    assert not audit["newly_eligible_rows"].any()


def test_replay_identity_audit_rejects_changed_saved_action() -> None:
    base = pd.DataFrame(
        {
            "field_season": ["field_2020"],
            "season": [2020],
            "issue_date": pd.to_datetime(["2020-07-01"]),
            "score": [0.3],
            "score_status": ["computed"],
            "message_issued": [False],
            "alarm_active": [False],
            "suppressed_repeat": [False],
            "action_reason": ["below_threshold"],
        }
    )
    passed = replay_identity_audit(
        base, base.copy(), model_code="C3", fold_id="test_2020"
    )
    assert passed["status"] == "passed"

    changed = base.copy()
    changed["message_issued"] = True
    with pytest.raises(AssertionError, match="identity failed"):
        replay_identity_audit(
            changed, base, model_code="C3", fold_id="test_2020"
        )
