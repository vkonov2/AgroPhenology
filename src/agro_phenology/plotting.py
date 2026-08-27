"""Plots used by the notebook; every function returns a Matplotlib figure."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


def plot_observation(
    daily: pd.DataFrame,
    threshold: float,
    predicted_date: date | None,
    observed_date: date,
    accumulation_start_date: date,
    output_path: str | Path | None = None,
):
    """Plot temperature, daily SET, and cumulative SET for one observation."""

    dates = pd.to_datetime(daily["date"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(dates, daily["temperature_mean_c"], color="#26734d")
    axes[0].set_ylabel("Температура, °C")
    axes[0].set_title("Среднесуточная температура")
    axes[1].bar(dates, daily["daily_degree_days_c_day"], color="#e19c24")
    axes[1].set_ylabel("Суточная СЭТ, °C·day")
    axes[2].plot(dates, daily["cumulative_degree_days_c_day"], color="#4c67ad")
    axes[2].axhline(threshold, color="black", linestyle="--", label=f"Порог {threshold:g} °C·day")
    axes[2].set_ylabel("Накопленная СЭТ, °C·day")
    axes[2].set_xlabel("Локальная дата")
    for axis in axes:
        axis.axvline(pd.Timestamp(accumulation_start_date), color="#777777", linestyle=":", label="Начало")
        axis.axvline(pd.Timestamp(observed_date), color="#c23b3b", linestyle="--", label="Наблюдение")
        if predicted_date is not None:
            axis.axvline(pd.Timestamp(predicted_date), color="#355cbd", linestyle="--", label="Расчёт")
        axis.grid(alpha=0.2)
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc="best")
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_sample_results(records: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save error distribution, date scatter, and hit-window plots."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    valid = records[records["calculation_status"] == "ok"].copy()
    paths: list[Path] = []
    if valid.empty:
        return paths

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid["error_days"], bins=min(15, max(3, len(valid))), color="#4c78a8", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--")
    ax.set(xlabel="Ошибка, дни (расчёт − наблюдение)", ylabel="Число расчётов", title="Распределение ошибки")
    path = destination / "error_distribution.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 7))
    observed = pd.to_datetime(valid["observed_date"])
    predicted = pd.to_datetime(valid["predicted_date"])
    ax.scatter(observed, predicted, c=valid["error_days"], cmap="coolwarm", edgecolor="black")
    limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
    ax.plot(limits, limits, "k--", label="Совпадение дат")
    ax.set(xlabel="Наблюдаемая дата", ylabel="Расчётная дата", title="Наблюдаемая дата против расчётной")
    ax.legend(); fig.autofmt_xdate()
    path = destination / "observed_vs_predicted.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fractions = [(valid["error_days"].abs() <= window).mean() for window in (3, 5, 7)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["±3 дня", "±5 дней", "±7 дней"], fractions, color="#59a14f")
    ax.set_ylim(0, 1); ax.set_ylabel("Доля расчётов"); ax.set_title("Попадание в временные окна")
    path = destination / "hit_windows.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def plot_late_blight_pilot_timeline(
    daily_features: pd.DataFrame,
    period_features: pd.DataFrame,
    observations: pd.DataFrame,
    output_path: str | Path,
):
    """Save one compact timeline for all pilot points and fixed observations."""

    required_daily = {"analysis_point_id", "date", "day_status"}
    required_period = {"observation_id", "period_end", "period_status"}
    if not required_daily.issubset(daily_features) or not required_period.issubset(period_features):
        raise ValueError("Late-blight plotting tables are missing required columns")
    daily = daily_features.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    periods = period_features.copy()
    periods["period_end"] = pd.to_datetime(periods["period_end"])
    point_order = list(dict.fromkeys(daily["analysis_point_id"].tolist()))
    date_order = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    status_codes = {"fail": 0.0, "indeterminate": 1.0, "pass": 2.0}
    matrix = np.full((len(point_order), len(date_order)), np.nan)
    date_index = {value: index for index, value in enumerate(date_order)}
    point_index = {value: index for index, value in enumerate(point_order)}
    for row in daily.itertuples(index=False):
        matrix[point_index[row.analysis_point_id], date_index[pd.Timestamp(row.date)]] = status_codes[row.day_status]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    x_values = mdates.date2num(date_order.to_pydatetime())
    x_edges = np.concatenate(([x_values[0] - 0.5], x_values + 0.5))
    y_edges = np.arange(len(point_order) + 1) - 0.5
    cmap = ListedColormap(["#d9d9d9", "#f2c94c", "#2e8b57"])
    heatmap = axes[0].pcolormesh(x_edges, y_edges, matrix, cmap=cmap, vmin=-0.5, vmax=2.5, shading="flat")
    axes[0].set_yticks(range(len(point_order)), point_order)
    axes[0].set_title(
        "Суточный статус Hutton: серый — FAIL, жёлтый — INDETERMINATE, "
        "зелёный — PASS, белый — вне периода"
    )
    axes[0].set_ylabel("Условная точка анализа")
    colorbar = fig.colorbar(heatmap, ax=axes[0], orientation="vertical", pad=0.01, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["FAIL", "INDETERMINATE", "PASS"])

    grouped = []
    for observation_id, group in periods.groupby("observation_id"):
        summary = group.groupby("period_end").agg(
            point_count=("analysis_point_id", "nunique"),
            pass_count=("period_status", lambda values: int((values == "pass").sum())),
            indeterminate_count=("period_status", lambda values: int((values == "indeterminate").sum())),
        )
        summary["pass_fraction"] = summary["pass_count"] / summary["point_count"]
        summary["indeterminate_fraction"] = summary["indeterminate_count"] / summary["point_count"]
        summary["observation_id"] = observation_id
        grouped.append(summary.reset_index())
    for summary in grouped:
        label = str(summary["observation_id"].iloc[0])
        axes[1].plot(summary["period_end"], summary["pass_fraction"], label=f"Hutton PASS: {label}")
        axes[1].plot(
            summary["period_end"], summary["indeterminate_fraction"], linestyle=":", alpha=0.8,
            label=f"Неопределено: {label}",
        )
    for observation in observations.itertuples(index=False):
        observed = pd.Timestamp(observation.observation_date)
        outcome_label = {"detected": "обнаружен", "not_detected": "не выявлен"}.get(
            observation.observation_outcome, observation.observation_outcome
        )
        axes[0].axvline(observed, color="#b22222", linestyle="--", linewidth=1.2)
        axes[1].axvline(observed, color="#b22222", linestyle="--", linewidth=1.2)
        axes[1].annotate(
            f"{observation.observation_id}: {outcome_label}",
            xy=(observed, 1.0), xytext=(3, -5), textcoords="offset points", rotation=90,
            va="top", ha="left", fontsize=8, color="#8b0000",
        )
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_ylabel("Доля точек")
    axes[1].set_xlabel("Локальная дата")
    axes[1].set_title("Доля условных точек с периодом Hutton и с неопределёнными данными")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper left", ncol=2, fontsize=8)
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    axes[1].set_xlim(date_order.min() - pd.Timedelta(days=2), date_order.max() + pd.Timedelta(days=5))
    fig.autofmt_xdate()
    fig.suptitle(
        "Пилот фитофтороза 2026: погодный индикатор Hutton\n"
        "Поляков: бутонизация 15–30.06, T/RH ERA5-Land, осадки ERA5",
        y=1.02,
    )
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    return fig
