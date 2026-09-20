from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from agro_phenology.apple_target_labels import (
    APPLE_SCAB_ID,
    CODLING_MOTH_ID,
    deduplicate_apple_visits,
    generic_absence,
    label_apple_rows,
    label_apple_scab,
    label_codling_moth,
    parse_detected_organisms,
    prepare_apple_target_visits,
)


def _entry(organism_id: int, detail: str) -> dict[str, object]:
    return {"organism_id": organism_id, "details_raw": detail}


def _row(
    observation_id: str,
    organisms: list[dict[str, object]] | str,
    *,
    day: str = "2025-07-10",
    field: str | None = "apple-field-1",
    crop_code: int = 41,
    generic: bool = False,
) -> dict[str, object]:
    encoded = organisms if isinstance(organisms, str) else json.dumps(organisms, ensure_ascii=False)
    return {
        "observation_id": observation_id,
        "observation_date": day,
        "crop_code": crop_code,
        "final_field_uid": field,
        "detected_organisms": encoded,
        "explicit_no_harmful_organisms": generic,
        "organisms_raw": "",
    }


def test_unlisted_target_is_unassessed_and_generic_absence_stays_distinct() -> None:
    ordinary = label_apple_scab([], generic_absent_flag=False)
    generic = label_apple_scab([], generic_absent_flag=True)
    codling = label_codling_moth([], generic_absent_flag=True)

    assert ordinary.status == "unassessed"
    assert not ordinary.listed_in_source
    assert generic.status == "generic_absent"
    assert not generic.listed_in_source
    assert codling.primary.status == "generic_absent"
    # A generic scouting statement does not prove that an adult trap was read.
    assert codling.adult_trap.status == "unassessed"


def test_codling_adult_zero_is_not_a_positive_larval_detection() -> None:
    labels = label_codling_moth(
        [_entry(CODLING_MOTH_ID, "Imago, Invāzijas pakāpe: 0 feromonu slazdā")]
    )

    assert labels.organism_listed_in_source
    assert labels.primary.status == "unassessed"
    assert not labels.primary.listed_in_source
    assert labels.adult_trap.status == "explicit_target_absent"
    assert labels.adult_trap.listed_in_source
    assert labels.adult_trap.value == 0


def test_codling_larva_and_adult_trap_are_labeled_independently() -> None:
    labels = label_codling_moth(
        [
            _entry(CODLING_MOTH_ID, "Kāpurs, Izplatība: 6.00%, Invāzijas pakāpe: 1 %"),
            _entry(CODLING_MOTH_ID, "Imago, Invāzijas pakāpe: 0 feromonu slazdā"),
        ]
    )

    assert labels.primary.status == "positive"
    assert labels.primary.value == 6
    assert labels.adult_trap.status == "explicit_target_absent"
    assert labels.adult_trap.value == 0


@pytest.mark.parametrize(
    ("detail", "expected_status", "expected_value"),
    [
        ("Kāpurs, Izplatība: 0.00%, Invāzijas pakāpe: 0 %", "explicit_target_absent", 0),
        ("Kāpurs, Izplatība: 2,50%, Invāzijas pakāpe: 1 %", "positive", 2.5),
        ("Kāpurs, Invāzijas pakāpe: 2 %", "unassessed", math.nan),
        ("Kāpurs, Invāzijas pakāpe: 0 feromonu slazdā", "unassessed", math.nan),
        ("Stadija, Izplatība: 5.00%", "unassessed", math.nan),
    ],
)
def test_codling_primary_requires_larval_prevalence(detail, expected_status, expected_value) -> None:
    label = label_codling_moth([_entry(CODLING_MOTH_ID, detail)]).primary
    assert label.status == expected_status
    if math.isnan(expected_value):
        assert math.isnan(label.value)
    else:
        assert label.value == expected_value


def test_codling_adult_parser_accepts_pheromone_trap_count() -> None:
    detail = "Imago, Invāzijas pakāpe: 1 (no 0 līdz 4) feromonu slazdā"
    label = label_codling_moth([_entry(CODLING_MOTH_ID, detail)]).adult_trap
    assert label.status == "positive"
    assert label.value == 1


def test_codling_glue_shield_is_not_the_frozen_pheromone_biofix() -> None:
    detail = "Imago, Invāzijas pakāpe: 11 uz līmes vairoga"
    label = label_codling_moth([_entry(CODLING_MOTH_ID, detail)]).adult_trap
    assert label.status == "unassessed"


def test_non_trap_imago_prevalence_does_not_become_trap_biofix() -> None:
    labels = label_codling_moth(
        [_entry(CODLING_MOTH_ID, "Imago, Izplatība: 2.00%, Invāzijas pakāpe: 0 %")]
    )
    assert labels.primary.status == "unassessed"
    assert labels.adult_trap.status == "unassessed"
    assert labels.organism_listed_in_source


@pytest.mark.parametrize(
    ("prevalence", "status"),
    [("0.00", "explicit_target_absent"), ("0,50", "positive"), ("24.00", "positive")],
)
def test_scab_uses_explicit_prevalence(prevalence, status) -> None:
    label = label_apple_scab(
        [_entry(APPLE_SCAB_ID, f"Izplatība: {prevalence}%, Attīstības pakāpe: 1%")]
    )
    assert label.status == status
    assert label.listed_in_source


def test_generic_absence_parser_handles_flag_and_source_text() -> None:
    assert generic_absence(True, "")
    assert generic_absence(False, "Kaitīgie organismi nav konstatēti")
    assert generic_absence(False, "Slimības netika konstatētas")
    assert not generic_absence(False, "Ābeļu kraupis - Izplatība: 2.00%")


def test_malformed_organism_json_is_auditable_and_not_a_positive() -> None:
    parsed, valid = parse_detected_organisms("not-json")
    assert parsed == []
    assert not valid

    rows = label_apple_rows(pd.DataFrame([_row("bad", "not-json", generic=True)]), APPLE_SCAB_ID)
    assert not bool(rows.iloc[0].organism_json_valid)
    assert rows.iloc[0].label_status == "generic_absent"
    assert not bool(rows.iloc[0].listed_in_source)


def test_label_rows_filters_non_apple_hosts() -> None:
    scab = _entry(APPLE_SCAB_ID, "Izplatība: 5.00%")
    frame = pd.DataFrame([_row("apple", [scab]), _row("potato", [scab], crop_code=166)])
    result = label_apple_rows(frame, APPLE_SCAB_ID)
    assert result.observation_id.tolist() == ["apple"]


def test_duplicate_priority_is_applied_without_mixing_adult_and_larval_targets() -> None:
    frame = pd.DataFrame(
        [
            _row(
                "unassessed",
                [_entry(CODLING_MOTH_ID, "Imago, Invāzijas pakāpe: 3 feromonu slazdā")],
            ),
            _row(
                "generic",
                [_entry(CODLING_MOTH_ID, "Imago, Invāzijas pakāpe: 0 feromonu slazdā")],
                generic=True,
            ),
            _row(
                "zero",
                [_entry(CODLING_MOTH_ID, "Kāpurs, Izplatība: 0.00%, Invāzijas pakāpe: 0 %")],
            ),
            _row(
                "positive",
                [_entry(CODLING_MOTH_ID, "Kāpurs, Izplatība: 8.00%, Invāzijas pakāpe: 2 %")],
            ),
        ]
    )
    visit = prepare_apple_target_visits(frame, CODLING_MOTH_ID).iloc[0]

    assert visit.label_status == "positive"
    assert visit.prevalence_pct == 8
    assert bool(visit.source_conflict)
    assert visit.adult_trap_status == "positive"
    assert visit.adult_trap_count == 3
    assert bool(visit.adult_trap_source_conflict)
    assert visit.source_row_count == 4


def test_scab_duplicate_priority_positive_over_zero_generic_and_unassessed() -> None:
    frame = pd.DataFrame(
        [
            _row("none", []),
            _row("generic", [], generic=True),
            _row("zero", [_entry(APPLE_SCAB_ID, "Izplatība: 0.00%")]),
            _row("positive", [_entry(APPLE_SCAB_ID, "Izplatība: 4.00%")]),
        ]
    )
    visit = prepare_apple_target_visits(frame, APPLE_SCAB_ID).iloc[0]
    assert visit.label_status == "positive"
    assert visit.prevalence_pct == 4
    assert bool(visit.source_conflict)
    assert visit.source_row_count == 4


def test_missing_field_ids_do_not_merge_unrelated_observations() -> None:
    positive = [_entry(APPLE_SCAB_ID, "Izplatība: 4.00%")]
    rows = label_apple_rows(
        pd.DataFrame([_row("first", positive, field=None), _row("second", positive, field=None)]),
        APPLE_SCAB_ID,
    )
    visits = deduplicate_apple_visits(rows, APPLE_SCAB_ID)
    assert len(visits) == 2
    assert visits.visit_id.nunique() == 2


def test_deduplicator_rejects_non_apple_rows() -> None:
    labeled = pd.DataFrame(
        {
            "crop_code": [166],
            "final_field_uid": ["field"],
            "observation_date": ["2025-07-10"],
            "label_status": ["unassessed"],
        }
    )
    with pytest.raises(ValueError, match="crop_code=41"):
        deduplicate_apple_visits(labeled, APPLE_SCAB_ID)
