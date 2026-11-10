"""Archive a bounded, explicitly incomplete corpus of Russian regional alerts.

These are regional presence reports and forecasts, not regular field visits.
Neither a missing report nor a weather warning creates a negative observation.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd

from .external_validation_io import DownloadError, write_json


SOURCE_ID = "rsc_regional_signals"
BASE_URL = "https://rosselhoscenter.ru"
MAP_URL = BASE_URL + "/search/map.php"
TARGETS = {
    "potato_late_blight": (r"фитофтор", "Phytophthora infestans"),
    "codling_moth": (r"яблон\w*\s+плодожорк", "Cydia pomonella"),
    "colorado_potato_beetle": (r"колорадск\w*\s+жук", "Leptinotarsa decemlineata"),
    "scab": (r"парш[аыиу]\b", "Venturia spp."),
    "onion_downy_mildew": (r"(?:пероноспороз|ложн\w*\s+мучнист\w*\s+рос)[^.!?]{0,40}лук", "Peronospora destructor"),
    "early_blight": (r"альтернариоз", "Alternaria spp."),
    "strawberry_grey_mould": (r"сер\w*\s+гнил|ботрит", "Botrytis cinerea"),
    "cabbage_maggot": (r"капустн\w*\s+мух", "Delia radicum"),
    "diamondback_moth": (r"капустн\w*\s+мол", "Plutella xylostella"),
    "carrot_fly": (r"морковн\w*\s+мух", "Chamaepsila rosae"),
}
MONTHS = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
          "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}
MONTH_PATTERN = "|".join(MONTHS)
NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](20\d{2}|\d{2})(?!\d)")
VERBAL_DATE = re.compile(rf"(?<!\d)(\d{{1,2}})\s+({MONTH_PATTERN})(?:\s+(20\d{{2}}))?", re.I)
OBSERVED = re.compile(r"\b(?:обнаружен[аоы]?|выявлен[аоы]?|отмечен[аоы]?|зафиксирован[аоы]?|начал(?:ся|ась|ось|ись))\b", re.I)
FORECAST = re.compile(r"\b(?:ожидается|ожидаются|прогнозируется|прогнозируются|возможна|возможно|вероятно|будет|будут)\b", re.I)
CONDITIONAL = re.compile(r"\b(?:при|в случае)\s+обнаружен|\b(?:не|не были|не было)\s+(?:обнаружен|выявлен|отмечен|зафиксирован)", re.I)
CROPS = {
    "potato": r"картофел|картофля", "tomato": r"томат|помидор", "apple": r"яблон",
    "pear": r"груш", "onion": r"\bлук[а-я]*\b", "strawberry": r"земляник|клубник",
    "cabbage": r"капуст", "carrot": r"морков", "rapeseed": r"рапс",
    "cucumber": r"огур", "sunflower": r"подсолнеч",
}


class PageParser(HTMLParser):
    """Small dependency-free parser of public Bitrix news pages and site map."""
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.body, self.heading, self.published, self.links = [], [], [], []
        self.active_link = None
        self.active_link_text = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if tag == "a":
            self.active_link = attributes.get("href")
            self.active_link_text = []
        if tag in {"p", "li", "br"} and any(row[1] == "body" for row in self.stack):
            self.body.append("\n")
        role = "body" if "news-detail__content" in classes else "date" if "news-detail__date" in classes else "heading" if tag == "h1" else None
        if tag not in self.VOID:
            self.stack.append((tag, role))

    def handle_data(self, text):
        if self.active_link:
            self.active_link_text.append(text)
        roles = {row[1] for row in self.stack}
        tags = {row[0] for row in self.stack}
        if tags & {"script", "style"}:
            return
        if "body" in roles:
            self.body.append(text)
        if "date" in roles:
            self.published.append(text)
        if "heading" in roles:
            self.heading.append(text)

    def handle_endtag(self, tag):
        if tag == "a" and self.active_link:
            self.links.append({"href": self.active_link, "title": " ".join(" ".join(self.active_link_text).split())})
            self.active_link = None
        if tag in {"p", "li"} and any(row[1] == "body" for row in self.stack):
            self.body.append("\n")
        for position in range(len(self.stack) - 1, -1, -1):
            if self.stack[position][0] == tag:
                del self.stack[position:]
                break


def parse_page(html: str) -> dict:
    parser = PageParser()
    parser.feed(html)
    return {"title": " ".join(" ".join(parser.heading).split()),
            "published_text": " ".join(" ".join(parser.published).split()),
            "body": "\n".join(" ".join(line.split()) for line in "".join(parser.body).splitlines() if line.strip()),
            "links": parser.links}


def _date_value(day, month, year):
    try:
        return date(int(year), int(month), int(day))
    except (ValueError, TypeError):
        return None


def stated_date(text: str, context: date | None = None) -> tuple[date | None, str]:
    """A missing year is contextual only within the previous 90 calendar days."""
    number = NUMERIC_DATE.search(text)
    if number:
        year = int(number[3])
        if year < 100:
            year += 2000
        return _date_value(number[1], number[2], year), "explicit_calendar_date"
    verbal = VERBAL_DATE.search(text)
    if verbal:
        month = MONTHS[verbal[2].lower()]
        if verbal[3]:
            return _date_value(verbal[1], month, verbal[3]), "explicit_calendar_date"
        if context:
            candidate = _date_value(verbal[1], month, context.year)
            if candidate is not None and 0 <= (context - candidate).days <= 90:
                return candidate, "year_from_report_context"
        return None, "month_day_only_year_unresolved"
    return None, "not_stated"


def _targets(title: str) -> list[str]:
    return [key for key, (pattern, _) in TARGETS.items() if re.search(pattern, title, re.I)]


def candidates_from_map(html: str) -> list[dict]:
    parser = PageParser()
    parser.feed(html)
    candidates = {}
    for row in parser.links:
        url = urljoin(BASE_URL, row["href"])
        parsed = urlparse(url)
        targets = _targets(row["title"])
        if parsed.netloc != "rosselhoscenter.ru" or not parsed.path.startswith("/ob-uchrezhdenii/filialy/") or not targets:
            continue
        if not re.search(r"сигнал|фитосанитар|мониторинг|обнаруж|выявлен", row["title"], re.I):
            continue
        # Exclude explicitly irrelevant crop adaptations; never relabel rape as cabbage.
        if targets == ["early_blight"] and re.search(r"рапс|подсолнеч", row["title"], re.I):
            continue
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 5:
            continue
        year = re.search(r"20(?:1[5-9]|2[0-6])", row["title"])
        candidates[url] = {"url": url, "title": row["title"], "targets": targets,
                           "publishing_branch": path_parts[3], "title_year": int(year[0]) if year else None}
    return sorted(candidates.values(), key=lambda row: row["url"])


def select_candidates(candidates: list[dict], limit: int = 150) -> list[dict]:
    """Spread the bounded corpus across targets and publishing region/year."""
    by_target = defaultdict(list)
    for row in candidates:
        by_target[row["targets"][0]].append(row)
    queues = {}
    for target, values in by_target.items():
        buckets = defaultdict(deque)
        for row in sorted(values, key=lambda r: (r["title_year"] or 0, r["url"])):
            buckets[(row["publishing_branch"], row["title_year"])].append(row)
        ordered = []
        while any(buckets.values()):
            for bucket in sorted(buckets, key=lambda k: (k[0], k[1] or 0)):
                if buckets[bucket]:
                    ordered.append(buckets[bucket].popleft())
        queues[target] = deque(ordered)
    selected = []
    while len(selected) < limit and any(queues.values()):
        for target in sorted(queues):
            if queues[target] and len(selected) < limit:
                selected.append(queues[target].popleft())
    return selected


def _filename(url: str, suffix: str = ".html") -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:20] + suffix


def _discover_rsc(client, max_reports: int = 150) -> list[dict]:
    prefix = f"raw/{SOURCE_ID}"
    source = {"source_id": SOURCE_ID, "provider": "Россельхозцентр",
              "title": "Региональные сигнализационные сообщения о болезнях и вредителях",
              "landing_url": MAP_URL, "access_status": "open",
              "license": {"id": None, "url": BASE_URL,
                          "notes": "Публичные сообщения учреждения; открытая лицензия на массовое перераспространение не установлена. Локальный исследовательский архив."},
              "evidence_kind": "regional_reports_presence_only",
              "crops": ["potato", "apple", "pear", "onion", "strawberry", "cabbage", "carrot"],
              "organisms": [value[1] for value in TARGETS.values()],
              "limitations": ["Неполный ограниченный корпус; отсутствие сообщения не означает отсутствие объекта.",
                              "Регион/публикующий филиал не является координатой или идентификатором поля.",
                              "Дата публикации, дата сообщения и явно названная дата наблюдения хранятся раздельно.",
                              "Прогноз и рекомендация не становятся подтверждённым наблюдением.",
                              "PDF-приложения архивируются отдельно, но автоматически не преобразуются в полевые исходы."], "files": []}
    try:
        robots = client.get_text(BASE_URL + "/robots.txt", prefix + "/discovery/robots.txt", max_bytes=200_000)
        if re.search(r"(?im)^Disallow:\s*(?:/|/search/|/search/map\.php)\s*$", robots):
            source.update(access_status="unavailable")
            source["limitations"].append("Текущая robots.txt запрещает чтение карты; обход не выполнялся.")
            return [source]
        site_map = client.get_text(MAP_URL, prefix + "/discovery/site_map.html", max_bytes=12_000_000)
    except (DownloadError, OSError) as error:
        source.update(access_status="unavailable")
        source["limitations"].append(str(error))
        return [source]
    candidates = candidates_from_map(site_map)
    selected = select_candidates(candidates, max_reports)
    write_json(client.root / prefix / "discovery/candidates.json", candidates)
    write_json(client.root / prefix / "discovery/selection.json", {"maximum_reports": max_reports, "selection_rule": "round_robin_target_region_and_title_year", "selected": selected})
    source["candidate_reports"] = len(candidates)
    source["selected_reports"] = len(selected)
    attachments, failures = {}, []
    for index, row in enumerate(selected, 1):
        name = _filename(row["url"])
        file = {"name": name, "url": row["url"], "role": "regional_report", "max_bytes": 2_000_000,
                "title": row["title"], "publishing_branch": row["publishing_branch"], "targets": row["targets"]}
        source["files"].append(file)
        try:
            html = client.get_text(row["url"], prefix + "/files/" + name, max_bytes=2_000_000)
            for link in parse_page(html)["links"]:
                url = urljoin(row["url"], link["href"])
                parsed = urlparse(url)
                if parsed.netloc == "rosselhoscenter.ru" and parsed.path.lower().endswith(".pdf") and parsed.path.startswith("/upload/"):
                    attachments[url] = {"name": _filename(url, ".pdf"), "url": url, "role": "attachment", "max_bytes": 10_000_000,
                                        "parent_report_url": row["url"], "title": link["title"]}
        except (DownloadError, OSError, ValueError) as error:
            failures.append({"url": row["url"], "error": str(error)})
        if index % 10 == 0 or index == len(selected):
            print(f"Россельхозцентр: прочитано {index}/{len(selected)} сообщений; найдено {len(attachments)} PDF", flush=True)
    source["files"].extend(attachments.values())
    source["discovery_failures"] = failures
    write_json(client.root / prefix / "discovery/attachments.json", list(attachments.values()))
    return [source]


def _fab_metadata(client) -> dict:
    """Archive harmless public metadata; do not bypass the protected CSV export."""
    source_id = "fight_against_blight_access_audit"
    base = "https://blight.hutton.ac.uk/"
    prefix = f"raw/{source_id}"
    source = {"source_id": source_id, "provider": "James Hutton Institute",
              "title": "Fight Against Blight — аудит доступа к истории вспышек",
              "landing_url": base, "access_status": "metadata_only",
              "license": {"id": None, "url": "https://www.hutton.ac.uk/terms",
                          "notes": "Условия выгрузки пригодной публичной выборки ещё требуют подтверждения."},
              "evidence_kind": "outbreak_register_access_audit",
              "crops": ["potato", "tomato"], "organisms": ["Phytophthora infestans"],
              "limitations": ["CSV-export /api/outbreaks/csv требует авторизацию (HTTP401); обход не выполнялся.",
                  "При проверке открытого JSON обнаружены isPublic=False и поля реальных координат/индекса; payload не архивирован.",
                  "Для включения нужен утверждённый публичный экспорт и документированная семантика дат dateSubmitted/dateReceived.",
                  "Наличие года в метаданных не доказывает полноту регулярного мониторинга или дату первого заражения."],
              "files": [{"name": "landing.html", "url": base, "role": "landing_page", "max_bytes": 1_000_000},
                        {"name": "years.json", "url": base + "api/outbreaks/years", "role": "metadata", "max_bytes": 100_000}]}
    try:
        client.get_text(base, prefix + "/files/landing.html", max_bytes=1_000_000)
        years = client.get_json(base + "api/outbreaks/years", prefix + "/files/years.json", max_bytes=100_000)
        if not isinstance(years, list) or not all(isinstance(year, int) for year in years):
            raise ValueError("Unexpected years metadata schema")
        source["listed_years"] = years
    except (DownloadError, OSError, ValueError) as error:
        source["limitations"].append(str(error))
    write_json(client.root / prefix / "discovery/access_audit.json", {
        "csv_export_requires_authorization": True,
        "outbreak_payload_archived": False,
        "private_coordinate_fields_archived": False,
        "available_metadata": source.get("listed_years", []),
        "required_next_input": "approved_public_export_with_outcome_date_definition",
    })
    return source


def discover(client, max_reports: int = 150) -> list[dict]:
    return _discover_rsc(client, max_reports=max_reports) + [_fab_metadata(client)]


def _statement_type(sentence: str) -> str:
    if OBSERVED.search(sentence) and not CONDITIONAL.search(sentence):
        return "observed"
    if FORECAST.search(sentence):
        return "forecast"
    return "ambiguous"


def normalize_report(html: str, source_url: str, *, fallback_title: str = "") -> tuple[dict, list[dict]]:
    parsed = parse_page(html)
    title = parsed["title"] or fallback_title
    published, _ = stated_date(parsed["published_text"])
    report_date, _ = stated_date(title)
    context = report_date or published
    target_keys = _targets(title)
    report_id = hashlib.sha256(source_url.encode()).hexdigest()[:24]
    parts = urlparse(source_url).path.strip("/").split("/")
    region = parts[3] if len(parts) >= 5 else None
    sentences = [s.strip() for s in re.split(r"\n+|(?<=[!?])\s+|(?<=\.)\s+(?=[А-Я])", parsed["body"]) if s.strip()]
    mentions = []
    for sentence in sentences:
        if len(sentence) < 20:
            continue
        statement_type = _statement_type(sentence)
        if statement_type == "ambiguous":
            continue
        explicit_targets = _targets(sentence)
        applicable = explicit_targets or (target_keys if len(target_keys) == 1 else [])
        for target in applicable:
            observation, precision = stated_date(sentence, context) if statement_type == "observed" else (None, "forecast_not_observation")
            # A dated legal reference inside a sentence must not backdate the finding.
            if observation and context and (observation > context or (context - observation).days > 180):
                observation, precision = None, "date_outside_report_context_requires_review"
            host_sentence = re.sub(TARGETS[target][0], " ", sentence, flags=re.I)
            crops = [name for name, pattern in CROPS.items() if re.search(pattern, host_sentence, re.I)]
            stage = "larval_emergence" if re.search(r"отрожден|выход\w* личин", sentence, re.I) else "oviposition_or_eggs" if re.search(r"яйцеклад|яиц|яйца", sentence, re.I) else "adult_flight" if re.search(r"л[её]т\w* бабоч|начал[а-я]* л[её]т|массов\w* л[её]т", sentence, re.I) else "larva" if re.search(r"личин|гусениц", sentence, re.I) else "adult" if re.search(r"имаго|из зимовки", sentence, re.I) else None
            organism = TARGETS[target][1]
            if target == "scab":
                organism = "Venturia inaequalis" if "apple" in crops or re.search(r"яблон", title, re.I) else "Venturia pyrina" if "pear" in crops or re.search(r"груш", title, re.I) else organism
            mentions.append({"mention_id": f"{report_id}:{len(mentions)+1}", "report_id": report_id,
                             "source_url": source_url, "published_date": published, "report_date": report_date,
                             "observation_date": observation, "interval_start": None, "interval_end": None,
                             "date_precision": precision, "region": region, "target_key": target,
                             "organism": organism, "crop": crops[0] if len(crops) == 1 else None, "stage": stage,
                             "crops_mentioned_json": json.dumps(crops, ensure_ascii=False),
                             "statement_type": statement_type, "evidence_excerpt": sentence[:1000],
                             "evidence_kind": "regional_report_not_field_visit", "absence_not_observed": True})
    observed = [m for m in mentions if m["statement_type"] == "observed"]
    forecast = [m for m in mentions if m["statement_type"] == "forecast"]
    dates = {m["observation_date"] for m in observed if m["observation_date"] is not None}
    report = {"report_id": report_id, "source_url": source_url, "title": title,
              "published_date": published, "published_at_text": parsed["published_text"], "report_date": report_date,
              "observation_date": next(iter(dates)) if len(dates) == 1 else None,
              "interval_start": None, "interval_end": None, "region": region,
              "geography_role": "publishing_branch_not_precise_observation_location",
              "organism": TARGETS[target_keys[0]][1] if len(target_keys) == 1 else None,
              "target_keys_json": json.dumps(target_keys, ensure_ascii=False), "crop": None,
              "statement_type": "observed" if observed else "forecast" if forecast else "ambiguous",
              "observed_mentions": len(observed), "forecast_mentions": len(forecast),
              "body_characters": len(parsed["body"]),
              "evidence_excerpt": (observed or forecast)[0]["evidence_excerpt"] if mentions else parsed["body"][:1000],
              "body_sha256": hashlib.sha256(parsed["body"].encode()).hexdigest(),
              "evidence_kind": "regional_report_not_field_visit"}
    return report, mentions


def normalize(source: dict, raw_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if source["source_id"] == "fight_against_blight_access_audit":
        summary = {"rows": 0, "schema": {}, "date_range": {"start": None, "end": None},
                   "endpoints": [], "limitations": source["limitations"], "output_files": ["normalization_audit.json"],
                   "raw_observations_collected": False, "metadata_only": True}
        write_json(output_dir / "normalization_audit.json", summary)
        return summary
    reports, mentions, failed = [], [], []
    for file in source.get("files", []):
        if file.get("role") != "regional_report":
            continue
        path = raw_dir / "files" / file["name"]
        if not path.is_file():
            failed.append({"file": file["name"], "reason": "download_unavailable"})
            continue
        try:
            report, records = normalize_report(path.read_text(encoding="utf-8-sig"), file["url"], fallback_title=file.get("title", ""))
            report["page_path"] = "files/" + file["name"]
            reports.append(report)
            mentions.extend(records)
        except (ValueError, UnicodeError) as error:
            failed.append({"file": file["name"], "reason": str(error)})
    report_frame = pd.DataFrame(reports)
    mention_frame = pd.DataFrame(mentions)
    report_frame.to_parquet(output_dir / "reports.parquet", index=False)
    mention_frame.to_parquet(output_dir / "mentions.parquet", index=False)
    date_values = [r["published_date"] for r in reports if r.get("published_date") is not None]
    summary = {"rows": len(reports), "mention_rows": len(mentions),
               "schema": {name: str(dtype) for name, dtype in report_frame.dtypes.items()},
               "date_range": {"start": str(min(date_values)) if date_values else None, "end": str(max(date_values)) if date_values else None},
               "regions": int(report_frame.region.nunique()) if len(report_frame) else 0,
               "dated_observed_mentions": sum(m["observation_date"] is not None and m["statement_type"] == "observed" for m in mentions),
               "statement_counts": mention_frame.statement_type.value_counts().to_dict() if len(mention_frame) else {},
               "endpoints": ["regional_reported_presence_or_stage", "regional_forecast_context_only"],
               "limitations": source.get("limitations", []) + ["Автоматические извлечения требуют ручной проверки перед включением в evaluation.",
                   "Дата без года разрешается только по контексту даты сообщения в пределах предыдущих90дней и помечается year_from_report_context.",
                   "Интервальные/неполные даты пока сохраняются в выдержке, но не превращаются в точный день.",
                   "PDF-приложения остаются исходными документами; машинныеmentionsполучены только изHTML."],
               "failures": failed, "output_files": ["reports.parquet", "mentions.parquet", "normalization_audit.json"]}
    write_json(output_dir / "normalization_audit.json", summary)
    return summary
