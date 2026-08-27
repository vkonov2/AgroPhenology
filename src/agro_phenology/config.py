"""Configuration objects and editable stage dictionaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class PhenologyModel:
    """Parameters of one pest-stage degree-day model."""

    pest_name: str
    target_stage_code: str
    target_stage_name: str
    base_temperature_c: float
    degree_day_threshold: float


CODLING_MOTH_MODELS: dict[str, PhenologyModel] = {
    "mass_larval_hatch": PhenologyModel(
        pest_name="яблонная плодожорка",
        target_stage_code="mass_larval_hatch",
        target_stage_name="массовое отрождение гусениц",
        base_temperature_c=10.0,
        degree_day_threshold=230.0,
    ),
    "mass_adult_flight": PhenologyModel(
        pest_name="яблонная плодожорка",
        target_stage_code="mass_adult_flight",
        target_stage_name="массовый лёт бабочек",
        base_temperature_c=10.0,
        degree_day_threshold=126.0,
    ),
}


DEFAULT_STAGE_ALIASES: dict[str, str] = {
    "массовое отрождение гусениц": "mass_larval_hatch",
    "массовый выход гусениц": "mass_larval_hatch",
    "отрождение гусениц": "mass_larval_hatch",
    "массовый лёт бабочек": "mass_adult_flight",
    "массовый лет бабочек": "mass_adult_flight",
}

KNOWN_NON_TARGET_STAGES = {
    "взрослая особь",
    "имаго",
    "лёт бабочек",
    "лет бабочек",
    "яйца",
    "единичная гусеница",
    "повреждение плодов",
    "вредитель обнаружен",
}


@dataclass
class ExperimentConfig:
    """Runtime settings kept separate from calculation functions."""

    model: PhenologyModel = field(default_factory=lambda: CODLING_MOTH_MODELS["mass_larval_hatch"])
    accumulation_start_date: date | None = None
    accumulation_start_rule: str = "observation_record_or_manual_global_date"
    daily_temperature_methods: tuple[str, ...] = ("hourly_mean", "min_max_mean")
    minimum_valid_hours: int = 20
    allow_incomplete_days: bool = False
    cache_dir: Path = Path("data/cache/open_meteo")
    results_dir: Path = Path("results")
    optional_hourly_variables: tuple[str, ...] = ()
    calculation_version: str = "0.1.0"
