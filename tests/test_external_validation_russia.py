from datetime import date

from agro_phenology.external_validation_russia import (
    _fab_metadata,
    candidates_from_map,
    normalize_report,
    parse_page,
    select_candidates,
    stated_date,
)


URL = "https://rosselhoscenter.ru/ob-uchrezhdenii/filialy/severo-zapadnyy/novgorodskaya-oblast/signalizatsionnoe-soobshchenie/"


def _page(title, published, body):
    return f'<h1 class="main-title">{title}</h1><div class="news-detail__date">{published}</div><div class="news-detail__content"><p>{body}</p></div><footer>Обнаружен вредитель 01.01.2000</footer>'


def test_separate_publication_and_contextual_observation_dates():
    report, mentions = normalize_report(_page("Сигнализационное сообщение №1 от 24.06.2025 г. Фитофтороз", "24 июня 2025, 11:30", "23 июня на полях картофеля отмечено проявление фитофтороза."), URL)
    assert report["published_date"] == date(2025, 6, 24)
    assert report["report_date"] == date(2025, 6, 24)
    assert report["observation_date"] == date(2025, 6, 23)
    assert mentions[0]["date_precision"] == "year_from_report_context"
    assert mentions[0]["crop"] == "potato"
    assert len(mentions) == 1


def test_forecast_is_not_positive_observation():
    report, mentions = normalize_report(_page("Фитофтороз картофеля", "24 июня 2025, 11:30", "В конце июня ожидается проявление фитофтороза картофеля."), URL)
    assert report["statement_type"] == "forecast"
    assert mentions[0]["observation_date"] is None
    assert mentions[0]["statement_type"] == "forecast"


def test_undated_presence_does_not_borrow_publication_day():
    report, mentions = normalize_report(_page("Фитофтороз картофеля", "24 июня 2025, 11:30", "На посадках картофеля обнаружен фитофтороз."), URL)
    assert report["published_date"] == date(2025, 6, 24)
    assert report["observation_date"] is None
    assert mentions[0]["date_precision"] == "not_stated"


def test_conditional_and_negative_are_not_confirmed_presence():
    for body in ["При обнаружении личинок колорадского жука рекомендован осмотр.", "При обследовании колорадский жук не обнаружен."]:
        report, mentions = normalize_report(_page("Колорадский жук", "24 июня 2025, 11:30", body), URL)
        assert report["statement_type"] == "ambiguous"
        assert not mentions


def test_cabbage_in_species_name_does_not_assign_host():
    _, mentions = normalize_report(_page("Капустная моль", "24 июня 2025, 11:30", "23 июня отмечен массовый лёт бабочек капустной моли на рапсе."), URL)
    assert mentions[0]["crop"] == "rapeseed"
    assert mentions[0]["stage"] == "adult_flight"


def test_year_context_not_resolved_to_future_or_remote_year():
    assert stated_date("23 декабря", date(2025, 1, 3))[0] is None
    assert stated_date("23 января", date(2025, 6, 3))[0] is None
    assert stated_date("23 июня 2024")[0] == date(2024, 6, 23)


def test_content_parser_preserves_only_article_and_links():
    parsed = parse_page(_page("Фитофтороз", "24 июня 2025", 'Отмечено развитие.<a href="/upload/a.pdf">Скачать</a>'))
    assert "2000" not in parsed["body"]
    assert parsed["links"] == [{"href": "/upload/a.pdf", "title": "Скачать"}]


def test_candidate_scope_and_selection_are_bounded_and_reproducible():
    html = "".join(f'<a href="/ob-uchrezhdenii/filialy/sibirskiy/region-{i}/signal-{i}/">Сигнализационное сообщение {i} Фитофтороз картофеля 2025</a>' for i in range(20))
    html += '<a href="https://example.org/fake">Сигнализационное сообщение Фитофтороз</a>'
    html += '<a href="/ob-uchrezhdenii/filialy/sibirskiy/region-a/signal-99/">Сигнализационное сообщение Альтернариоз рапса</a>'
    candidates = candidates_from_map(html)
    assert len(candidates) == 20
    assert len(select_candidates(candidates, 7)) == 7
    assert select_candidates(candidates, 7) == select_candidates(list(reversed(candidates)), 7)


def test_fab_archive_never_requests_private_outbreak_payload(tmp_path):
    class Client:
        root = tmp_path
        calls = []

        def get_text(self, url, path, **kwargs):
            self.calls.append(url)
            return "<html>Public landing</html>"

        def get_json(self, url, path, **kwargs):
            self.calls.append(url)
            return [2024, 2025, 2026]

    client = Client()
    source = _fab_metadata(client)
    assert source["access_status"] == "metadata_only"
    assert client.calls == ["https://blight.hutton.ac.uk/", "https://blight.hutton.ac.uk/api/outbreaks/years"]
    assert all(file["role"] in {"landing_page", "metadata"} for file in source["files"])
    audit = (tmp_path / "raw/fight_against_blight_access_audit/discovery/access_audit.json").read_text()
    assert '"outbreak_payload_archived": false' in audit
