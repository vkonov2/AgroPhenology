"""Fixed numerical degree-day formulas against the archived UCANR challenge.

This is an arithmetic benchmark, not validation of any biological endpoint.
Formulas are expressed in whatever consistent temperature unit is supplied;
the frozen challenge uses degrees Fahrenheit and thresholds 50/70 F.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .external_validation_io import digest_file, utc_now, write_json


CONTRACT = Path("docs/research/external_validation/dd_numerical_contract.json")
DEFAULT_RAW = Path("data/external_validation/20261110_external_v2/raw/ucanr_degree_day_challenge/files")
CURVES = {"sine": "sine", "tri": "triangle"}
CUTOFFS = {"horiz": "horizontal", "vert": "vertical", "intrmd": "intermediate"}
METHODS = [f"{prefix}{shape}_{cut}" for prefix in ("sng", "dbl") for shape in CURVES for cut in CUTOFFS]
COMPARISON_TOLERANCES = (.0050001, .01, .1)


def _component(tmin: np.ndarray, tmax: np.ndarray, threshold: float, curve: str) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean excess and fraction strictly above one threshold."""
    mean = .5 * (tmin + tmax)
    area = np.zeros(tmin.shape, dtype=float)
    fraction = np.zeros(tmin.shape, dtype=float)
    entirely_above = tmin >= threshold
    area[entirely_above] = mean[entirely_above] - threshold
    fraction[entirely_above] = 1.0
    # A constant temperature AT the threshold is not strictly above it.
    fraction[(tmin == threshold) & (tmax == threshold)] = 0.0
    crossing = (tmin < threshold) & (tmax > threshold)
    lo, hi, center = tmin[crossing], tmax[crossing], mean[crossing]
    if curve == "sine":
        amplitude = .5 * (hi-lo)
        angle = np.arcsin((threshold-center)/amplitude)
        fraction[crossing] = (.5*np.pi-angle)/np.pi
        area[crossing] = ((center-threshold)*(.5*np.pi-angle)+amplitude*np.cos(angle))/np.pi
    else:
        fraction[crossing] = (hi-threshold)/(hi-lo)
        area[crossing] = (hi-threshold)**2/(2*(hi-lo))
    return area, fraction


def estimate_degree_days(tmin, tmax, lower: float = 50, upper: float = 70, *,
                         curve: str = "sine", cutoff: str = "horizontal",
                         double: bool = False, next_min=None) -> np.ndarray:
    """Analytic sine/triangle integration; double needs the NEXT minimum.

    The intermediate cutoff follows the documented UCIPM routines: its extra
    removal cannot exceed that of the vertical cutoff. Components for double
    methods are averaged before this restriction is applied. No formulas or
    tolerances are selected by agreement with the reference observations.
    """
    if curve not in {"sine", "triangle"} or cutoff not in CUTOFFS.values():
        raise ValueError("Unknown curve or cutoff")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("Thresholds must be finite and increasing")
    lo, hi = np.broadcast_arrays(np.asarray(tmin, dtype=float), np.asarray(tmax, dtype=float))
    if double:
        if next_min is None:
            following = np.full(lo.shape, np.nan)
        else:
            lo, hi, following = np.broadcast_arrays(lo, hi, np.asarray(next_min, dtype=float))
    valid = np.isfinite(lo) & np.isfinite(hi)
    if (valid & (lo > hi)).any():
        raise ValueError("Daily minimum exceeds maximum")
    if double:
        valid &= np.isfinite(following)
        if (valid & (following > hi)).any():
            raise ValueError("Next minimum exceeds the preceding maximum")
    result = np.full(lo.shape, np.nan)
    if not valid.any():
        return result
    lo_valid, hi_valid = lo[valid], hi[valid]
    lower_area, _ = _component(lo_valid, hi_valid, lower, curve)
    upper_area, upper_fraction = _component(lo_valid, hi_valid, upper, curve)
    if double:
        lower_next, _ = _component(following[valid], hi_valid, lower, curve)
        upper_next, fraction_next = _component(following[valid], hi_valid, upper, curve)
        lower_area = .5*(lower_area+lower_next)
        upper_area = .5*(upper_area+upper_next)
        upper_fraction = .5*(upper_fraction+fraction_next)
    horizontal = lower_area-upper_area
    removal = (upper-lower)*upper_fraction
    if cutoff == "horizontal":
        values = horizontal
    elif cutoff == "vertical":
        values = horizontal-removal
    else:
        values = horizontal-np.minimum(upper_area, removal)
    result[valid] = np.maximum(values, 0.0)
    return result


def _compute(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather = pd.read_csv(raw/"data__espartoa-weather-2020.csv")
    answers = pd.read_csv(raw/"data__ucipm_results__ucipm_low50_high70_all.csv")
    weather["date"] = pd.to_datetime(weather.date, format="%Y-%m-%d")
    answers["date"] = pd.to_datetime(answers.date, format="%m/%d/%Y")
    if weather.date.duplicated().any() or answers.date.duplicated().any():
        raise ValueError("Duplicate reference dates")
    expected = pd.date_range("2020-01-01", "2020-12-31")
    if not pd.DatetimeIndex(weather.date).equals(expected) or not pd.DatetimeIndex(answers.date).equals(expected):
        raise ValueError("Reference dates do not cover the frozen year exactly")
    if not weather[["tmin", "tmax"]].equals(answers[["tmin", "tmax"]]):
        raise ValueError("Reference answer temperatures differ from weather")
    next_min = weather.tmin.shift(-1).to_numpy(dtype=float)
    rows, metrics = [], []
    for method in METHODS:
        prefix_shape, cut = method.split("_")
        double = prefix_shape.startswith("dbl")
        shape = prefix_shape[3:]
        prediction = estimate_degree_days(weather.tmin, weather.tmax, curve=CURVES[shape],
                                          cutoff=CUTOFFS[cut], double=double, next_min=next_min)
        reference = answers[method].to_numpy(dtype=float)
        computable = np.isfinite(prediction)
        differences = prediction[computable]-reference[computable]
        for date, pred, target in zip(weather.date, prediction, reference):
            rows.append(dict(method=method, date=date, predicted_f_days=pred, reference_f_days=float(target),
                             computable=bool(np.isfinite(pred)),
                             missing_reason="next_day_minimum_not_in_input" if pd.isna(pred) else ""))
        metric = dict(method=method, source="ucanr_degree_day_challenge", reference_days=len(weather),
                      computable_days=int(computable.sum()), abstention_days=int((~computable).sum()),
                      mae_f_days=float(np.mean(np.abs(differences))), max_abs_error_f_days=float(np.max(np.abs(differences))),
                      signed_error_sum_f_days=float(differences.sum()),
                      predicted_total_common_f_days=float(prediction[computable].sum()),
                      reference_rounded_daily_total_common_f_days=float(reference[computable].sum()))
        for tol in COMPARISON_TOLERANCES:
            key = "half_rounding_unit" if tol < .01 else str(tol).replace(".", "_")
            metric[f"within_{key}_days"] = int((np.abs(differences) <= tol).sum())
        metrics.append(metric)
    return pd.DataFrame(rows), pd.DataFrame(metrics)


def _report(metrics: pd.DataFrame, root: Path) -> str:
    lines = ["# Численная проверка методов суммы температур", "", "10 ноября 2026. Это проверка арифметики, **не биологическая валидация**: в эталоне нет наблюдений вредителя/болезни и нельзя получить событийное покрытие, precision, AP или качество предупреждений.", "",
             "Источник — [UCANR-IGIS Degree Day Challenge](https://github.com/UCANR-IGIS/degree-day-challenge/tree/1766f1c8f53ab32d7dd023bba9a748d555351a04), MIT, закреплённый commit `1766f1c8f53ab32d7dd023bba9a748d555351a04`. Погода: 366 дней 2020 года; значения эталона получены авторами от UC IPM. Пороги 50 и 70 °F, единица результата °F·дни. Для перехода к °C·дням результат умножается на 5/9, пороги и температуры также переводятся согласованно.", "",
             "Реализованы аналитические single/double sine и triangle с horizontal/vertical/intermediate cutoff. Формулы следуют [определениям UC IPM](https://ipm.ucanr.edu/weather/ddconcepts.html) и [документированным вычислительным процедурам](https://ipm.ucanr.edu/weather/ddroutines.html). Для intermediate дополнительное вычитание ограничено vertical removal; у double сначала усредняются компоненты двух полусуток. Разногласия с эталоном сохраняются, а не исправляются подбором формулы.", "",
             "Для double нужен минимум следующего дня. Его нет для 31 декабря, поэтому последний день шести double-вариантов не вычисляется: ни повторения последнего минимума, ни заполнения из будущего источника. Single сравнивается на 366 днях, double на 365. Это 4 386 сопоставимых значений из 4 392 опубликованных. Double нельзя применять к текущим завершившимся суткам до получения следующего минимума.", "",
             "Допуски зафиксированы до полного сравнения. Половина единицы округления эталона — 0,0050001 °F·дня (небольшой запас лишь для float); дополнительно показываются заранее выбранные 0,01 и 0,1. Колонки показывают совпадение чисел, а не долю правильно предсказанных случаев вредителя.", "",
             "| Метод | Сопоставимых дней | MAE, °F·дня | Max ошибка | В пределах 0,0050001 | В пределах 0,01 | В пределах 0,1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in metrics.itertuples(index=False):
        lines.append(f"| {r.method} | {r.computable_days} | {r.mae_f_days:.6f} | {r.max_abs_error_f_days:.6f} | {r.within_half_rounding_unit_days}/{r.computable_days} | {r.within_0_01_days}/{r.computable_days} | {r.within_0_1_days}/{r.computable_days} |")
    lines += ["", "Расхождения особенно важны у промежуточного/вертикального ограничения: опубликованный численный эталон и описанная формула не приравниваются друг к другу автоматически. Ни официальный badge, ни прохождение формальной сертификации не заявляются. Погодные данные одного года не являются независимыми полевыми опытами.", "",
              "Старая `degree_days.daily_degree_days` рассчитывает max(средняя температура−base,0) без верхнего порога. У неё нет совпадающего эталонного столбца этого challenge. Поэтому отсутствие её в таблице не является скрытым исключением неудобного результата: приравнять средний метод к sine/triangle было бы неверно. Старый модуль и прежние исследования не менялись.", "",
              "Суммы в машинной таблице считаются на общих днях из опубликованных округлённых суточных значений. Они не выдаются за внутреннюю неокруглённую накопленную сумму калькулятора. Новые формулы не подставлялись в замороженные CPB/Luke-модели: биологические пороги зависят от выбранного метода суммы температур.", "",
              "## Воспроизведение", "", "`predictions.parquet/csv` содержит все 4 392 комбинации дата × метод, включая шесть явных отказов; `metrics.csv` — отдельные метрики методов. Контракт, хеши входов/выходов, snapshot кода и README источника сохранены в локальном запуске.", "", "```bash",
              f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_dd_benchmark verify --output-dir {root}",
              "PYTHONPATH=src .venv/bin/python -m agro_phenology.external_dd_benchmark run --output-dir results/external_validation/ucanr_degree_day_challenge/reproduction_NEW",
              "PYTHONPATH=src .venv/bin/python -m pytest tests/test_external_dd_benchmark.py -q", "```", "",
              "Следующий приоритет — использовать совместимую, проверенную численную реализацию только там, где пороги биологической модели заданы для того же метода. Численный benchmark не заменяет полевых исходов.", ""]
    return "\n".join(lines)


def run(raw: Path, root: Path) -> dict:
    if root.exists():
        raise FileExistsError("Benchmark output is immutable; use a new directory")
    inputs = [raw/"data__espartoa-weather-2020.csv", raw/"data__ucipm_results__ucipm_low50_high70_all.csv", raw/"README.md", raw/"LICENSE", CONTRACT, Path(__file__)]
    hashes = {str(p.resolve()): digest_file(p) for p in inputs}
    root.mkdir(parents=True)
    shutil.copyfile(CONTRACT, root/"contract.json")
    for p in [Path(__file__), raw/"README.md", raw/"LICENSE"]:
        destination = root/"source_snapshot"/p.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, destination)
    predictions, metrics = _compute(raw)
    for name, frame in [("predictions", predictions), ("metrics", metrics)]:
        frame.to_parquet(root/f"{name}.parquet", index=False)
        frame.to_csv(root/f"{name}.csv", index=False)
    (root/"REPORT_RU.md").write_text(_report(metrics, root), encoding="utf-8")
    if any(digest_file(Path(p)) != h for p, h in hashes.items()):
        raise RuntimeError("Inputs changed during calculation")
    inventory = {str(p.relative_to(root)): digest_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root/"manifest.json", dict(status="complete", purpose="numerical_not_biological_validation", created_at=utc_now(), raw_dir=str(raw.resolve()), input_hashes=hashes, file_inventory=inventory,
               python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__))
    return dict(status="complete", output_dir=str(root), methods=len(metrics), comparable_values=int(predictions.computable.sum()), unavailable_values=int((~predictions.computable).sum()), metrics=metrics.to_dict("records"))


def verify(root: Path) -> dict:
    manifest = json.loads((root/"manifest.json").read_text())
    failures = [p for p, h in manifest["file_inventory"].items() if not (root/p).is_file() or digest_file(root/p) != h]
    inputs = [p for p, h in manifest["input_hashes"].items() if not Path(p).is_file() or digest_file(Path(p)) != h]
    checked = []
    if not failures and not inputs:
        computed = _compute(Path(manifest["raw_dir"]))
        for name, frame in zip(["predictions", "metrics"], computed):
            try:
                pd.testing.assert_frame_equal(frame, pd.read_parquet(root/f"{name}.parquet"), check_exact=True)
                checked.append(name)
            except AssertionError:
                failures.append("recompute:"+name)
    return dict(status="passed" if manifest["status"] == "complete" and not failures and not inputs else "failed", checked_files=len(manifest["file_inventory"]), checked_inputs=len(manifest["input_hashes"]), recomputed_tables=checked, mismatches=failures, input_mismatches=inputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.raw_dir, args.output_dir) if args.command == "run" else verify(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
