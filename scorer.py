"""
scorer.py - Основной модуль скоринга кандидатов по бизнес-доменам (v7-hybrid).

Алгоритм v7-hybrid:
  1. Запросные домены кодируются через EXPANSIONS_V6 (ручной, стабильный словарь).
  2. Кандидатские домены ищутся в emb_v7 (авто-расширенные embeddings, 71% покрытие).
  3. Для каждого запросного термина - наилучший match среди доменов кандидата (cosine).
  4. Бонусы: таксономия L1/L2 + co-occurrence.
  5. Штраф за неполное покрытие: score × (0.5 + 0.5 × hit_rate).
  6. Бонус за нишевую концентрацию (+0.08, если ≥70% доменов кандидата совпадают по L1).
  7. Итоговый score нормализуется в [0, 1] (деление на 1.28).
  8. Запросные домены фильтруются: filter_query_domains() / _QUERY_NOISE_EXACT.

Использование как модуль:
    from scorer import rank
    results = rank(["финтех", "интеграция банковского ядра"])
    for r in results[:5]:
        print(f"#{r['id']}  {r['score']:.3f}  hits={r['hit_rate']:.0%}")

Использование как CLI:
    python scorer.py "финтех" "интеграция банковского ядра"
    python scorer.py --top 10 "kafka" "микросервисы"
    python scorer.py --json "финтех" | python -m json.tool
"""

import sys, os, json, math, re, unicodedata, warnings, argparse
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"          # работаем с кэшем, без сети
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / ".venv_libs"))

import io, contextlib
_stderr_buf = io.StringIO()

with contextlib.redirect_stderr(_stderr_buf):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────
#  ПАРАМЕТРЫ СКОРИНГА
# ──────────────────────────────────────────────────────────────
SIM_THRESHOLD  = 0.72   # порог cosine-сходства для «попадания»
L1_BONUS       = 0.10   # бонус за совпадение отрасли (L1 таксономии)
L2_BONUS       = 0.20   # бонус за совпадение подотрасли (L2)
COOCCUR_WEIGHT = 0.15   # вес co-occurrence бонуса
NICHE_CONC_MIN = 0.30   # минимальная концентрация для применения L1/L2 бонусов
NICHE_BONUS    = 0.08   # бонус за высокую нишевую концентрацию (≥70%)

# Минимальная длина query-домена (символов) - фильтр случайных строк/опечаток
QUERY_MIN_LEN  = 3

# Точные совпадения (после _norm) - типичный мусор в запросах, даёт ложные попадания в emb
_QUERY_NOISE_EXACT = frozenset({
    "abc", "xyz", "qwe", "asd", "zxc", "rty", "fgh", "vbn", "mnb", "iop", "jkl",
    "qwerty", "asdfgh", "asdf", "zxcv", "zxcvbn", "zxcvbnm", "wasd",
    "test", "foo", "bar", "baz", "lol", "wow", "nan", "nil",
    "xxx", "yyy", "zzz", "aaa", "qqq", "йцукен", "фывапр", "ячсмить",
})


# Символы-варианты дефиса → стандартный дефис
_DASH_MAP = {"\u2011": "-", "\u2012": "-", "\u2013": "-",
             "\u2014": "-", "\u2015": "-", "\u2212": "-"}


# ──────────────────────────────────────────────────────────────
#  РУЧНЫЕ РАСШИРЕНИЯ ЗАПРОСНЫХ ДОМЕНОВ (v6 - проверенный словарь)
# ──────────────────────────────────────────────────────────────
EXPANSIONS = {
    # Финансы - общее
    "финтех":                   "финтех финансовые технологии fintech платёжные сервисы онлайн-финансы банковский сектор",
    "банкинг":                  "банкинг банковский сектор банковские услуги финансовые операции banking",
    # Банковские протоколы и интеграции (SWIFT-блок)
    "swift-месседжинг":         "swift-месседжинг SWIFT MT МТ-сообщения SWIFT gpi ISO 20022 межбанковские переводы "
                                "MX-сообщения финансовый месседжинг банковские сообщения interbank messaging "
                                "корреспондентский банкинг SWIFT network финансовые операции трансграничные платежи",
    "банковские интеграции":    "банковские интеграции интеграция АБС core banking integration API банка "
                                "SWIFT интеграция межбанковское взаимодействие open banking финансовые API "
                                "ЦФТ Diasoft Flextera интеграция платёжных систем",
    "интеграция банковского ядра": "интеграция банковского ядра АБС core banking ЦФТ Diasoft Flextera "
                                "SWIFT интеграция банковские API межбанковские протоколы "
                                "финансовые системы legacy banking modernization",
    "платёжные системы":        "платёжные системы SWIFT СБП НСПК VISA Mastercard MIR платёжная инфраструктура "
                                "эквайринг процессинг межбанковские расчёты payment systems clearing settlement",
    "межбанковские расчёты":    "межбанковские расчёты корреспондентские счета SWIFT RTGS клиринг "
                                "ностро-счета лоро-счета межбанковские переводы settlement clearing interbank",
    "импортозамещение банковских продуктов":
                                "импортозамещение банковских продуктов российский софт АБС ЦФТ Diasoft "
                                "замена SWIFT альтернативы западным банковским системам СПФС CIPS финтех",
    "e-commerce":               "e-commerce электронная коммерция интернет-торговля онлайн-магазин маркетплейс",
    "интернет-торговля":        "интернет-торговля e-commerce электронная коммерция онлайн-магазин",
    "электронная коммерция":    "электронная коммерция e-commerce интернет-торговля онлайн-ритейл",
    "adtech":                   "adtech рекламные технологии programmatic advertising dsp ssp rtb",
    "рекламные технологии":     "рекламные технологии adtech programmatic dsp ssp rtb реклама в интернете",
    "медицина":                 "медицина здравоохранение медицинские информационные системы клиника",
    "логистика":                "логистика управление цепочками поставок склад доставка транспорт supply chain",
    "туризм":                   "туризм travel-tech онлайн-бронирование путешествия авиабилеты отели",
    "crm":                      "crm управление взаимоотношениями с клиентами customer relationship management",
    "биллинг":                  "биллинг выставление счетов billing платёжные операции тарификация",
    "страхование":              "страхование страховые продукты insurance страховые выплаты полис",
    "ритейл":                   "ритейл розничная торговля retail продажи магазин",
    # QA / Тестирование
    "тестирование программного обеспечения":
                                "тестирование программного обеспечения software testing QA quality assurance "
                                "тест тестировщик проверка ПО automated testing manual testing qa engineer "
                                "управление качеством верификация валидация test plan test case",
    "обеспечение качества программного обеспечения":
                                "обеспечение качества программного обеспечения QA software quality assurance "
                                "тестирование контроль ПО quality gates управление качеством релизов "
                                "qa-процессы верификация test plan regression smoke testing приёмочное тестирование",
    "автоматизация тестирования":
                                "автоматизация тестирования test automation QA автотесты CI/CD тестировщик "
                                "selenium pytest appium robot framework автоматизированное тестирование "
                                "cypress playwright e2e тестирование api-тестирование postman",
    "управление дефектами":     "управление дефектами defect management bug tracking баг-трекер jira issues "
                                "тестирование репорт бага lifecycle bugs qa",
    "управление инцидентами":   "управление инцидентами incident management ITSM сбои аварии helpdesk support "
                                "on-call мониторинг alerting",
    "управление качеством релизов":
                                "управление качеством релизов release management QA gate релизный процесс "
                                "deployment обеспечение качества ПО quality gates release candidate staging",
    "контроль качества электроники":
                                "контроль качества электроники electronics QA hardware testing verification "
                                "pcb testing inspection",
    "api-тестирование":         "api-тестирование api testing REST API postman swagger тестирование бэкенда "
                                "integration testing contract testing",
    "валидация данных":         "валидация данных data validation data quality тестирование данных проверка "
                                "корректности данных ETL testing обеспечение качества данных",
    "тестирование и обеспечение качества по":
                                "тестирование и обеспечение качества по QA software testing quality assurance "
                                "автоматизация тестирования верификация приёмочное тестирование",
    "тестирование и обеспечение качества цифровых сервисов":
                                "тестирование и обеспечение качества цифровых сервисов QA digital testing "
                                "веб-тестирование мобильное тестирование performance testing",
    "sql-аналитика":            "sql-аналитика SQL аналитика баз данных data analysis запросы отчётность BI "
                                "data engineering ETL datawarehouse",
    # Информационная безопасность
    "информационная безопасность":
                                "информационная безопасность кибербезопасность ИБ защита данных cybersecurity "
                                "information security pentest penetration testing SIEM SOC DLP IAM ФСТЭК ЦБ РФ "
                                "ISO 27001 PCI DSS vulnerability assessment incident response",
    "кибербезопасность":        "кибербезопасность информационная безопасность ИБ cybersecurity security engineer "
                                "penetration testing SIEM SOC защита от атак threat intelligence",
    # DevOps / Инфраструктура
    "облачные сервисы":         "облачные сервисы cloud services AWS Azure GCP облачная инфраструктура SaaS PaaS IaaS "
                                "kubernetes docker контейнеризация serverless микросервисы",
    "управление проектами":     "управление проектами project management PM agile scrum kanban "
                                "product owner scrum master планирование спринтов roadmap стейкхолдеры",
    # Игры / Геймдев
    "монетизация игр":          "монетизация игр game monetization freemium in-app purchases микротранзакции мобильные игры",
    "игровая монетизация":      "игровая монетизация game monetization freemium покупки внутри игры мобильные",
    "игровые механики":         "игровые механики game mechanics геймдизайн gameplay loop progression мобильные игры",
    "игровая механика":         "игровая механика game mechanics геймплей игровой процесс дизайн",
    "геймдизайн":               "геймдизайн game design игровые механики нарратив уровни персонажи",
    "геймдев":                  "геймдев gamedev разработка игр game development unity unreal engine",
    "геймдев платформа":        "геймдев платформа game development platform unity unreal разработка игр",
    "разработка видеоигр":      "разработка видеоигр video game development gamedev игровой движок",
    "игровая платформа":        "игровая платформа gaming platform онлайн-игры мультиплеер монетизация",
    "онлайн-игровая платформа": "онлайн-игровая платформа online gaming platform мультиплеер монетизация игровые механики",
    "игровая индустрия":        "игровая индустрия game industry монетизация игровые механики game monetization геймдизайн мобильные игры",
    "видеоигры":                "видеоигры video games геймдев игровая индустрия консольные мобильные монетизация",
    "мобильные игры":           "мобильные игры mobile games монетизация игровые механики in-app purchases казуальные игры game monetization",
    "игровые услуги":           "игровые услуги gaming services онлайн-игры монетизация игровые платформы",
    "онлайн-беттинг":           "онлайн-беттинг online betting gambling ставки на спорт игровая платформа мобильные игры монетизация",
    "гэмблинг":                 "гэмблинг gambling казино онлайн ставки игорный бизнес",
    "gamefi лаунчпад":          "gamefi лаунчпад blockchain games play-to-earn nft gaming web3",
    "маркетплейс цифровых товаров": "маркетплейс цифровых товаров digital goods marketplace игровые предметы nft",
    # Блокчейн / Web3 / DeFi
    "blockchain":               "blockchain блокчейн технология распределённого реестра distributed ledger "
                                "смарт-контракты smart contracts DeFi NFT Web3 криптовалюта ethereum solidity "
                                "токенизация децентрализованные приложения dApps",
    "смарт-контракты":          "смарт-контракты smart contracts solidity ethereum blockchain DeFi автоматизация "
                                "контракты в блокчейне web3 токены protocol",
    "децентрализованные финансы": "децентрализованные финансы DeFi decentralized finance blockchain "
                                "протоколы ликвидности yield farming стейкинг DEX AMM криптовалюта",
    "блокчейн":                 "блокчейн blockchain распределённый реестр смарт-контракты ethereum "
                                "криптовалюта токенизация web3 DeFi NFT decentralized",
    "блокчейн-услуги":          "блокчейн-услуги blockchain services смарт-контракты web3 DeFi "
                                "токенизация distributed ledger криптовалюта",
    "web3":                     "web3 blockchain децентрализованные приложения DeFi NFT смарт-контракты "
                                "криптовалюта ethereum solidity токенизация",
    "nft":                      "nft non-fungible token токен blockchain маркетплейс цифровые активы web3",
    "криптовалюта":             "криптовалюта cryptocurrency bitcoin ethereum blockchain цифровые активы токены",
    "деfi":                     "defi децентрализованные финансы decentralized finance blockchain смарт-контракты "
                                "yield farming стейкинг DEX ликвидность",
    "децентрализованная финансовая платформа":
                                "децентрализованная финансовая платформа DeFi decentralized finance "
                                "blockchain смарт-контракты DEX ликвидность yield web3",
    "токенизированный nft-маркетплейс":
                                "токенизированный nft-маркетплейс nft маркетплейс blockchain токены "
                                "digital assets web3 ethereum",
    # Мобильная разработка
    "ios разработка":           "ios разработка iOS swift objective-c apple мобильная разработка "
                                "xcode appstore iphone ipad mobile development",
    "android разработка":       "android разработка android kotlin java mobile development "
                                "google play мобильная разработка приложения смартфон",
    "разработка мобильных приложений":
                                "разработка мобильных приложений mobile development iOS android swift kotlin "
                                "react native flutter cross-platform мобильное приложение",
    "мобильные приложения":     "мобильные приложения mobile apps iOS android swift kotlin flutter "
                                "react native приложение смартфон",
    "кроссплатформенная разработка":
                                "кроссплатформенная разработка cross-platform flutter react native xamarin "
                                "ios android мобильная разработка",
    # Медтех / Телемедицина
    "телемедицина":             "телемедицина telemedicine дистанционная медицина онлайн-консультации "
                                "здравоохранение eHealth mHealth медицинские сервисы digital health",
    "мобильное здравоохранение": "мобильное здравоохранение mHealth mobile health медицинские приложения "
                                "телемедицина digital health здоровье носимые устройства",
    "электронные медкарты":     "электронные медкарты electronic health records EHR EMR медицинские "
                                "информационные системы МИС ЕГИСЗ электронная медицинская карта HL7 FHIR",
    "медицинские информационные системы":
                                "медицинские информационные системы МИС ЕГИСЗ HL7 FHIR EHR EMR "
                                "цифровое здравоохранение eHealth телемедицина клиническая система",
    "здравоохранение":          "здравоохранение healthcare медицина телемедицина digital health МИС "
                                "eHealth медицинские технологии клиника hospital",
    "медтех":                   "медтех medtech медицинские технологии digital health телемедицина "
                                "МИС мобильное здравоохранение wearables healthtech",
    # AdTech / Programmatic (запросные синонимы)
    "programmatic":             "programmatic реклама programmatic advertising DSP SSP RTB аукцион "
                                "рекламные технологии adtech цифровая реклама",
    "dsp платформа":            "dsp demand-side platform programmatic реклама DSP RTB аукцион adtech",
    "rtb аукцион":              "rtb real-time bidding аукцион programmatic реклама DSP SSP adtech",
    "программматическая реклама": "программматическая реклама programmatic advertising DSP SSP RTB "
                                "adtech аукцион цифровая реклама",
    "рекламная платформа":      "рекламная платформа ad platform programmatic DSP SSP RTB adtech",
    # Стриминг / медиа
    "стриминг":                 "стриминг streaming видеостриминг live streaming видео-контент OTT "
                                "медиаплатформа онлайн-видео VOD",
    "стриминг видео":           "стриминг видео video streaming OTT VOD live streaming медиаплатформа",
    # Электронные платежи
    "электронные платежи":      "электронные платежи electronic payments e-payments платёжные системы "
                                "онлайн-оплата банковские карты мобильные платежи",
    # Онлайн-торговля/ритейл (синонимы)
    "онлайн-ритейл":            "онлайн-ритейл online retail e-commerce интернет-магазин маркетплейс",
    "маркетплейс":              "маркетплейс marketplace e-commerce онлайн-торговля платформа продаж",
    # Страхование (синонимы)
    "страховой сектор":         "страховой сектор insurance страхование insurtech страховые продукты",
    "страховые онлайн-услуги":  "страховые онлайн-услуги online insurance insurtech страхование digital insurance",
    # Телеком
    "телеком":                  "телеком телекоммуникации telecommunications связь мобильный оператор "
                                "интернет-провайдер BSS OSS telecom",
    "телекоммуникационные услуги": "телекоммуникационные услуги telecommunications telecom BSS OSS мобильный оператор",
}


# ──────────────────────────────────────────────────────────────
#  ЗАГРУЗКА ДАННЫХ (ленивая - при первом вызове rank())
# ──────────────────────────────────────────────────────────────
_model       = None
_candidates  = None   # {id: [domain, ...]}
_cooccur     = None   # {domain: {domain: count}}
_taxonomy    = None   # {domain: {l1, l2}}
_domain_list = None   # список всех доменов (порядок = строки emb_v7)
_emb_v7      = None   # np.ndarray shape (N, 384), L2-нормированные
_d2i         = None   # {domain: index in _emb_v7}
_exp_auto    = None   # авто-расширения из expansions.json

# Слияние синонимичных L1 после перегенерации taxonomy.json (LLM даёт разные названия одной отрасли).
# Формат: {"как в файле": "каноническое имя"}. Пусто - текущий taxonomy.json уже согласован (23 L1).
_TAXONOMY_L1_REMAP: dict[str, str] = {}


def _apply_taxonomy_l1_remap() -> None:
    """Applies optional L1 canonical mapping to loaded taxonomy in-place.

    Input:
      - Uses global `_taxonomy` dict and `_TAXONOMY_L1_REMAP` mapping.
    Output:
      - None. Updates `_taxonomy` values in memory.
    """
    global _taxonomy
    if not _taxonomy or not _TAXONOMY_L1_REMAP:
        return
    for d, meta in list(_taxonomy.items()):
        l1 = meta.get("l1", "")
        if l1 in _TAXONOMY_L1_REMAP:
            _taxonomy[d] = {**meta, "l1": _TAXONOMY_L1_REMAP[l1]}


def _load():
    """Инициализирует все ресурсы. Вызывается один раз при первом rank()."""
    global _model, _candidates, _cooccur, _taxonomy
    global _domain_list, _emb_v7, _d2i, _exp_auto

    if _model is not None:
        return  # уже загружено

    with contextlib.redirect_stderr(_stderr_buf):
        _model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder=str(BASE / ".model_cache"),
        )

    _candidates = {
        int(k): v for k, v in
        json.loads((BASE / "candidates_normalized.json").read_text()).items()
    }
    _cooccur  = json.loads((BASE / "cooccur_matrix.json").read_text())
    _taxonomy = json.loads((BASE / "taxonomy.json").read_text()) \
                if (BASE / "taxonomy.json").exists() else {}
    # При необходимости слить синонимичные L1 (после обновления taxonomy.json LLM-ом)
    _apply_taxonomy_l1_remap()

    npz = np.load(BASE / "domain_embeddings_v7.npz", allow_pickle=True)
    _domain_list = list(npz["domains"])
    _emb_v7      = npz["embeddings"]
    _d2i         = {d: i for i, d in enumerate(_domain_list)}

    # Авто-расширения из expansions.json (ручные EXPANSIONS имеют приоритет)
    _exp_auto = {}
    exp_file = BASE / "expansions.json"
    if exp_file.exists():
        raw = json.loads(exp_file.read_text())
        for k, v in raw.items():
            clean = re.sub(r"\[src:\w+\]", "", v).strip()
            _exp_auto[_norm(k)] = clean

    # Объединяем: авто + ручные (ручные приоритетнее)
    _exp_auto.update({_norm(k): v for k, v in EXPANSIONS.items()})


# ──────────────────────────────────────────────────────────────
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ──────────────────────────────────────────────────────────────
def _norm(label: str) -> str:
    """Нормализует метку домена: Unicode NFC, унификация дефисов, lower, strip."""
    label = unicodedata.normalize("NFC", label)
    for ch, repl in _DASH_MAP.items():
        label = label.replace(ch, repl)
    return re.sub(r"\s+", " ", label.lower()).strip()


def _is_plausible_query_domain(nd: str) -> bool:
    """False для заведомо бессмысленных query-доменов (клавиатурный мусор, тестовые токены)."""
    if len(nd) < QUERY_MIN_LEN:
        return False
    if nd in _QUERY_NOISE_EXACT:
        return False
    return True


def filter_query_domains(domains: list[str]) -> list[str]:
    """Публичный фильтр для вызовов до rank (live_demo, score_file)."""
    return [d for d in domains if _is_plausible_query_domain(_norm(d))]


def _encode(domains: list[str]) -> tuple[np.ndarray, list[str]]:
    """Кодирует запросные домены только через ручной словарь EXPANSIONS.

    Hybrid-стратегия: запросы - стабильный ручной словарь, кандидаты - emb_v7.
    Это защищает от семантического дрейфа при авто-расширении запросов.
    """
    normed = [_norm(d) for d in domains]
    texts  = [EXPANSIONS.get(d, d) for d in normed]   # только ручные расширения
    with contextlib.redirect_stderr(_stderr_buf):
        embs = _model.encode(texts, normalize_embeddings=True)
    return embs, normed


def _query_l1s(q_norm: list[str]) -> set[str]:
    """Множество L1-категорий для запросных доменов (если есть в таксономии)."""
    return {_taxonomy[d]["l1"] for d in q_norm if d in _taxonomy}


def _concentration(cand_domains: list[str], query_l1s: set[str]) -> float:
    """Доля доменов кандидата, совпадающих по L1 с запросом."""
    if not cand_domains or not query_l1s:
        return 0.0
    m = sum(1 for d in cand_domains
            if d in _taxonomy and _taxonomy[d]["l1"] in query_l1s)
    return m / len(cand_domains)


def _tax_bonus(q_dom: str, c_dom: str,
               cand_domains: list[str], query_l1s: set[str]) -> float:
    """Таксономический бонус: +L2_BONUS при совпадении подотрасли, +L1_BONUS при совпадении отрасли.
    Применяется только если концентрация кандидата ≥ NICHE_CONC_MIN."""
    ta = _taxonomy.get(q_dom)
    tb = _taxonomy.get(c_dom)
    if not ta or not tb:
        return 0.0
    if _concentration(cand_domains, query_l1s) < NICHE_CONC_MIN:
        return 0.0
    if ta["l2"] == tb["l2"]:
        return L2_BONUS
    if ta["l1"] == tb["l1"]:
        return L1_BONUS
    return 0.0


def _score_from_full_sim(full_sim: np.ndarray, q_norm: list[str],
                         cand_domains: list[str], query_l1s: set[str],
                         weights: dict | None) -> dict:
    """Вычисляет score одного кандидата, используя предвычисленную матрицу full_sim.

    full_sim: (Q, V) - сходство каждого запросного домена со всеми V доменами словаря,
              вычислено один раз в rank() для всего словаря.
    weights:  опциональный dict {query_domain: float} - важность каждого запросного домена
              (2.0 = обязательно, 1.0 = нейтрально, 0.6 = желательно).
    """
    known_names, known_idxs = [], []
    for d in cand_domains:
        if d in _d2i:
            known_names.append(d)
            known_idxs.append(_d2i[d])
    if not known_names:
        return {"score": 0.0, "hit_rate": 0.0, "details": []}

    Q = len(q_norm)
    # Срез уже вычисленной матрицы: (Q, K) - только домены этого кандидата
    cs      = full_sim[:, known_idxs]
    best_k  = np.argmax(cs, axis=1)          # (Q,) - лучший домен кандидата на каждый запрос
    sims    = cs[np.arange(Q), best_k]        # (Q,) - соответствующие сходства

    details, raw_scores, weight_list = [], [], []
    for qi in range(Q):
        qd  = q_norm[qi]
        sim = float(sims[qi])
        cd  = known_names[best_k[qi]]
        co  = _cooccur.get(qd, {}).get(cd, 0)
        co_b = math.log1p(co) / math.log1p(5) if co else 0.0
        tax  = _tax_bonus(qd, cd, cand_domains, query_l1s)
        w    = (weights.get(qd, 1.0) if weights else 1.0)
        raw_sc = w * ((sim + COOCCUR_WEIGHT * co_b + tax) if sim >= SIM_THRESHOLD
                      else sim * 0.30)
        details.append({
            "query_domain": qd,
            "match":        cd,
            "similarity":   round(sim, 4),
            "hit":          sim >= SIM_THRESHOLD,
            "tax_bonus":    round(tax, 3),
            "weight":       round(w, 2),
        })
        raw_scores.append(raw_sc)
        weight_list.append(w)

    w_total  = sum(weight_list) or 1.0
    hit_rate = sum(1 for d in details if d["hit"]) / max(Q, 1)
    # Взвешенное среднее с soft coverage penalty
    score    = (sum(raw_scores) / w_total) * (0.5 + 0.5 * hit_rate)

    if _concentration(cand_domains, query_l1s) >= 0.70:
        score += NICHE_BONUS

    # Нормализуем score в [0, 1]: теоретический максимум без бонусов = 1.0 (sim=1, hit=1),
    # с tax L2_BONUS=0.20 + niche=0.08 → max ~ 1.28. Делим на 1.28 → [0, 1].
    score = min(score / 1.28, 1.0)

    return {"score": round(score, 4), "hit_rate": hit_rate, "details": details}


# ──────────────────────────────────────────────────────────────
#  ПУБЛИЧНОЕ API
# ──────────────────────────────────────────────────────────────
def rank(query_domains: list[str],
         weights: dict[str, float] | None = None,
         top: int = None) -> list[dict]:
    """Ранжирует всех кандидатов по совпадению бизнес-доменов с запросом.

    Args:
        query_domains: список строковых доменов запроса вакансии.
        weights:       опциональный dict {domain: float} - важность домена
                       (2.0 = обязательно, 1.0 = нейтрально, 0.6 = желательно).
                       Генерируется через live_demo.estimate_domain_weights().
        top:           если задан - возвращает только top-N кандидатов.

    Returns:
        Список словарей, отсортированный по score по убыванию.
    """
    _load()
    query_domains = filter_query_domains(query_domains)
    if not query_domains:
        return []

    q_embs, q_norm = _encode(query_domains)
    q_l1s = _query_l1s(q_norm)

    # Ключевая оптимизация: (Q, V) матрица сходства вычисляется один раз,
    # вместо per-candidate вызовов cosine_similarity. Ускорение ~10-20x.
    full_sim = (q_embs @ _emb_v7.T).astype(np.float32)  # (Q, V)

    # Нормализуем ключи weights
    w_norm = {_norm(k): v for k, v in weights.items()} if weights else None

    results = []
    for cid, cdoms in _candidates.items():
        s = _score_from_full_sim(full_sim, q_norm, cdoms, q_l1s, w_norm)
        results.append({"id": cid, "domains": cdoms, **s})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top] if top else results


def explain(candidate_id: int, query_domains: list[str],
            weights: dict[str, float] | None = None) -> dict:
    """Возвращает детальное объяснение score для одного кандидата."""
    _load()
    query_domains = filter_query_domains(query_domains)
    if not query_domains:
        raise ValueError("После фильтрации не осталось ни одного допустимого домена в запросе")
    q_embs, q_norm = _encode(query_domains)
    q_l1s = _query_l1s(q_norm)
    cdoms = _candidates.get(candidate_id)
    if cdoms is None:
        raise ValueError(f"Кандидат #{candidate_id} не найден")
    full_sim = (q_embs @ _emb_v7.T).astype(np.float32)
    w_norm = {_norm(k): v for k, v in weights.items()} if weights else None
    s = _score_from_full_sim(full_sim, q_norm, cdoms, q_l1s, w_norm)
    return {"id": candidate_id, "domains": cdoms, **s}


def explain_text(result: dict) -> str:
    """Читаемое объяснение score на русском - без LLM, на основе шаблонов.

    Примеры вывода:
      "Хорошее соответствие (67% требований). Есть: тестирование ПО, автоматизация.
       Не найдено: sql-аналитика."
    """
    score    = result["score"]
    hit_rate = result["hit_rate"]
    details  = result.get("details", [])
    hits     = [d for d in details if d["hit"]]
    misses   = [d for d in details if not d["hit"]]

    # Вердикт по нормализованному score [0..1]
    # 0.78+  → Отличное (соответствует ~1.0 в старой шкале)
    # 0.51+  → Хорошее  (соответствует ~0.65+)
    # 0.35+  → Частичное
    # 0.16+  → Слабое
    if score >= 0.78:
        verdict = "Отличное соответствие"
    elif score >= 0.51:
        verdict = "Хорошее соответствие"
    elif score >= 0.35:
        verdict = "Частичное соответствие"
    elif score >= 0.16:
        verdict = "Слабое соответствие"
    else:
        verdict = "Не соответствует"

    parts = [f"{verdict} ({hit_rate:.0%} требований покрыто)"]

    if hits:
        # Приоритет - домены с высоким весом (обязательные)
        sorted_hits = sorted(hits, key=lambda d: (-d.get("weight", 1.0), -d["similarity"]))
        names = [d["match"] for d in sorted_hits[:3]]
        parts.append(f"Есть: {', '.join(names)}")

    if misses:
        # Показываем пропущенные с высоким весом первыми
        sorted_miss = sorted(misses, key=lambda d: -d.get("weight", 1.0))
        names = [d["query_domain"] for d in sorted_miss[:2]]
        parts.append(f"Не найдено: {', '.join(names)}")

    if not hits:
        parts.append("Нет совпадений по заданным доменам")

    return ". ".join(parts)


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────
def _bar(v: float, width: int = 18) -> str:
    """Builds a compact ASCII score bar for CLI output.

    Input:
      - v: normalized score in [0, 1].
      - width: bar width in characters.
    Output:
      - Formatted string with bar and numeric score.
    """
    # score нормализован в [0, 1]
    norm = min(max(v, 0.0), 1.0)
    filled = int(norm * width)
    ch   = "█" if norm >= 0.85 else ("▓" if norm >= 0.70 else ("▒" if norm >= 0.55 else "░"))
    return "[" + ch * filled + "·" * (width - filled) + f"] {v:.3f}"


def _print_results(results: list[dict], query_domains: list[str],
                   verbose: bool = False, n: int = 10) -> None:
    """Prints top ranked candidates in readable CLI table.

    Input:
      - results: ranking output from `rank()`.
      - query_domains: original query domain labels.
      - verbose: if True, prints per-domain details.
      - n: number of top rows to print.
    Output:
      - None. Writes formatted text to stdout.
    """
    W = 72
    q_str = "  +  ".join(query_domains)
    print(f"\n  Запрос: {q_str}")
    print("  " + "─" * (W - 2))
    print(f"  {'Место':<6} {'#':<6} {'hit%':>5}   {'Score':<25}  Лучший match")
    print("  " + "─" * (W - 2))

    for i, r in enumerate(results[:n], 1):
        best = max(r["details"], key=lambda d: d["similarity"]) if r["details"] else {}
        hit  = "✓" if best.get("hit") else "✗"
        tx   = f" +tax" if best.get("tax_bonus", 0) > 0 else ""
        match_info = (f"{hit} '{best['query_domain'][:14]}' → '{best['match'][:16]}'"
                      f" {best['similarity']:.3f}{tx}") if best else ""
        print(f"  {i:<6} #{r['id']:<5} {r['hit_rate']:>4.0%}   {_bar(r['score'])}  {match_info}")

        if verbose:
            for d in r["details"]:
                fl = "✓" if d["hit"] else "·"
                tx = f" +tax={d['tax_bonus']:.2f}" if d["tax_bonus"] > 0 else ""
                print(f"         {fl}  '{d['query_domain'][:24]:<26}→ '{d['match'][:22]}'  {d['similarity']:.3f}{tx}")

    print("  " + "─" * (W - 2))


def main() -> None:
    """CLI entry point for domain scorer.

    Input:
      - Command-line arguments (domains, top, verbose, json, explain).
    Output:
      - None. Prints ranking or explanation and exits with code 0/1.
    """
    parser = argparse.ArgumentParser(
        description="Scorer - ранжирование кандидатов по бизнес-доменам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python scorer.py "финтех" "интеграция банковского ядра"
  python scorer.py --top 5 "kafka" "микросервисы" "event-driven"
  python scorer.py --verbose "автоматизация тестирования" "QA"
  python scorer.py --json "e-commerce" "маркетплейс"
  python scorer.py --explain 22 "автоматизация тестирования" "QA"
""",
    )
    parser.add_argument("domains", nargs="+", help="Домены запроса вакансии")
    parser.add_argument("--top",     type=int, default=10, help="Показать топ-N (по умолчанию 10)")
    parser.add_argument("--verbose", action="store_true",  help="Детали по каждому домену")
    parser.add_argument("--json",    action="store_true",  help="Вывод в JSON")
    parser.add_argument("--explain", type=int, metavar="ID",
                        help="Детальное объяснение для конкретного кандидата")
    args = parser.parse_args()

    if args.explain is not None:
        result = explain(args.explain, args.domains)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"\n  Кандидат #{result['id']}  score={result['score']:.3f}  hit_rate={result['hit_rate']:.0%}")
            print(f"  Домены: {result['domains']}")
            print()
            for d in result["details"]:
                fl = "✓" if d["hit"] else "✗"
                tx = f"  +tax={d['tax_bonus']:.2f}" if d["tax_bonus"] > 0 else ""
                print(f"  {fl}  '{d['query_domain']}'  →  '{d['match']}'  sim={d['similarity']:.3f}{tx}")
        return

    results = rank(args.domains)
    if args.json:
        print(json.dumps(results[:args.top], ensure_ascii=False, indent=2))
    else:
        _print_results(results, args.domains, verbose=args.verbose, n=args.top)


if __name__ == "__main__":
    main()
