# Исследовательский контур ранних предупреждений

Контур работает только с первой зарегистрированной записью фитофтороза картофеля в поле-сезоне. Он строит ежедневные решения на 08:00 `Europe/Riga`, использует past-only признаки с отсечением `issue_date - 2 дня`, симулирует сообщения и оценивает окно 3–10 календарных дней. Реальные сообщения не отправляются.

Главный завершённый запуск первого цикла: `results/late_blight_early_warning/20260910_first_cycle_v3`. Он заменяет v2 только как итоговый проверенный снимок: после независимой рецензии добавлен настоящий контроль непрерывной тревоги и машинно уточнён критерий сопоставимой нагрузки. V2 сохранён неизменным. Исходные таблицы, модели и журналы содержат полевые идентификаторы и остаются локальными. Русский агрегированный отчёт не содержит идентификаторов и координат.

Завершённый второй цикл: `results/late_blight_early_warning/20260910_second_cycle_v4`. Он добавляет диагностику разных попаданий старых политик, C6 как погодную CatBoost-поправку к logits C0, калибровочный и календарный контроли, сервисный fallback к C0 и строгую вложенную временную проверку. Каталоги `20260910_second_cycle_v1`–`v3` — промежуточные аудитные снимки; итоговые выводы и команды находятся в v4. Научные таблицы, журналы и модели остаются локальными.

Завершённый третий цикл: `results/late_blight_early_warning/20260910_third_cycle_v2`. Он не обучает новые модели и не меняет alpha: на замороженных score проверяет повторное сообщение при существенном росте score во время cooldown, старую политику, её перенастройку и контроль с укороченным cooldown. `20260910_third_cycle_v1` — сохранённый предрезультатный failed-снимок; научные результаты в нём не создавались после выявления неоднозначного представления fallback.

Завершён этап готовности к проспективному теневому сбору:
`results/late_blight_early_warning/20260910_shadow_readiness_v2`. Он фиксирует
модели и политики, проверяет входы на дату решения, хранит исходные ответы,
решения и наблюдения раздельно и воспроизводит состояние из журнала. Выполнена
только offline-техническая демонстрация; живой сбор, сеть, реальные уведомления
и расписание не запускались. Спецификация входов и исходов находится в
`prospective_logging_spec.md`, замороженный контракт — в
`shadow_readiness_contract.json`, а операционные команды — в `RUNBOOK.md`
итогового запуска.

На 10 сентября 2026 года погодная поправка frozen C6 операционно невычислима:
ей нужна ERA5 до `issue_date−2`, тогда как документированная задержка ERA5T
составляет около пяти дней. C6 поэтому может работать только как точный C0
fallback в общей истории политики, а C4/C5 должны воздерживаться. Для живого
запуска также отсутствуют актуальный реестр подключённых картофельных полей и
независимый от score процесс полевых наблюдений.

`20260910_shadow_readiness_v1` сохранён как предрезультатный failed-снимок:
повторный replay обнаружил побайтовое изменение SQLite при чтении. В v2 чтение
переведено в read-only режим, а SHA-256 базы проверяется до и после replay и
`verify-log`.

## Окружение

Фактически проверенные версии зафиксированы в `requirements-early-warning.txt` и в `execution_manifest.json` запуска:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  -r requirements-early-warning.txt -e .
```

Команда установки и CLI проверены 10 сентября 2026 года. Диапазоны `.[early-warning]` в `pyproject.toml` удобны для нового окружения, но не заменяют список фактически проверенных версий.

## Запуск

Малый сквозной контроль с одной временной свёрткой и одной пробой Optuna:

```bash
.venv/bin/agro-late-blight-early-warning run \
  --contract docs/research/late_blight_early_warning/evaluation_contract.json \
  --run-id smoke_reproduction_01 --smoke
```

Полный цикл с новым уникальным идентификатором:

```bash
.venv/bin/agro-late-blight-early-warning run \
  --contract docs/research/late_blight_early_warning/evaluation_contract.json \
  --run-id full_reproduction_01
```

Второй цикл с новым уникальным идентификатором:

```bash
.venv/bin/agro-late-blight-cycle2 run \
  --contract docs/research/late_blight_early_warning/cycle2_evaluation_contract.json \
  --v3-run results/late_blight_early_warning/20260910_first_cycle_v3 \
  --run-id <new_unique_cycle2_run_id>
```

Третий цикл с новым уникальным идентификатором:

```bash
.venv/bin/python -m agro_phenology.early_warning_cycle3_pipeline run \
  --contract docs/research/late_blight_early_warning/cycle3_evaluation_contract.json \
  --v3-run results/late_blight_early_warning/20260910_first_cycle_v3 \
  --v4-run results/late_blight_early_warning/20260910_second_cycle_v4 \
  --run-id <new_unique_cycle3_run_id>
```

Третий цикл использует сохранённые прогнозы и не переобучает модели. Идентификатор нового запуска должен указывать на отсутствующий или пустой каталог.

Фактический итоговый запуск `20260910_first_cycle_v3` выполнен установленной консольной командой выше. Идентификатор завершённого запуска нельзя использовать повторно: непустой run-каталог не перезаписывается.

Отчёт из сохранённых прогнозов и журналов создаётся в новом соседнем каталоге; завершённый run не меняется:

```bash
.venv/bin/agro-late-blight-early-warning report \
  --run-dir results/late_blight_early_warning/20260910_first_cycle_v3
```

Продолжение сохранённых Optuna-study также копирует их в новый соседний каталог:

```bash
.venv/bin/agro-late-blight-early-warning resume-optuna \
  --run-dir results/late_blight_early_warning/20260910_first_cycle_v3 \
  --additional-trials-per-fold 2
```

Продолжение study само по себе не пересчитывает внешний test. Для сравнения новой конфигурации нужен новый полный run с заранее зафиксированным бюджетом.

## Основные артефакты

- `data_inventory.json`, `data_audit.json`, `input_version_comparison.json` — наличие, схемы, хеши, происхождение и потери;
- `field_seasons.parquet`, `events.parquet`, `daily_decisions.parquet` — событийный реестр, цензурирование и ежедневные даты решений;
- `predictions.parquet`, `notification_log.parquet`, `alarm_states.parquet` — out-of-sample score, доступность входов и последовательные решения;
- `event_metrics.csv`, `burden_metrics.csv`, `paired_comparisons.csv`, `budget_grid_pooled.csv` — событийные counts, нагрузка и парные сравнения;
- `optuna/`, `optuna_trials.csv`, `optuna_seed_checks.csv`, `models/`, `policy_selection.csv` — ограниченный поиск, модели и политика;
- `source_snapshot.json`, `execution_manifest.json`, `test_results.json` — точный снимок незакоммиченного кода, окружение, хеши и проверки;
- `report_ru.md` — полный русский отчёт первого цикла.

Во втором цикле дополнительно сохранены `v3_event_intersections.csv`, `v3_miss_reason_summary.csv`, `v3_yearly_funnel.csv`, периодические и Polyakov-диагностики, `oof_predictions.parquet`, `oof_provenance.csv`, `c6_raw_predictions.parquet`, `policy_selection.csv`, `paired_annual_metrics.csv`, `paired_year_bootstrap.csv`, `leave_one_year_out.csv`, `gained_lost_summary.csv`, `message_change_summary.csv`, 84 модельных пакета и русский `report_ru.md`. Полный список с SHA-256 находится в `20260910_second_cycle_v4/execution_manifest.json`.

В третьем цикле дополнительно сохранены `audit_population_and_suppression.md`, исправленная eligible-таблица gained/lost, `frozen_scores.parquet`, проверки воспроизведения score и P0, `validation_policy_candidates.csv`, `policy_selection.csv`, `predictions.parquet`, `notification_log.parquet`, `alarm_states.parquet`, `event_hits.parquet`, годовые и агрегированные метрики, `gained_lost_summary.csv`, bootstrap, leave-one-year-out, русский `report_ru.md`, тесты и manifest. В v2 проверены 43 выходных хеша, 87 файлов v3, 307 файлов v4 и 18 файлов снимка исходников.

Ключевой результат третьего цикла: P_growth для погодной C6 в полном сервисе дала 62/87 своевременных событий против 46/87 у старой политики, но повысила нагрузку до 1,798 сообщения/30 и 39,9% тревожных дней. P_short дала 64/87 при ещё большей нагрузке. Из 132 growth-сообщений 79 относятся к активному C6 weather, 51 — к `alpha=0` с эффективным C0 и два — к C0 fallback. На общей маске результаты C6 равны 49/87 для P_growth, 45/87 для P0 и 57/87 для P_short; C0/C1/C4/C5/календарное окно с P_growth дали 46/46/53/53/45 из 87. Погодная специфичность и заранее заданный сильный эффект не подтверждены. Подробный разбор 127/87, 357 изменённых score и годовой неоднородности находится в `STATUS.md` и `20260910_third_cycle_v2/report_ru.md`.

Метрики всех трёх циклов, ограничения и следующий приоритет записаны в `STATUS.md`. Спецификация будущего теневого проспективного журнала находится в `prospective_logging_spec.md`.
