"""
build_expansions.py - Автоматическое расширение словаря доменов.

Берём все уникальных домена из candidates_normalized.json и для каждого
строим «расширенный» текст, который подаётся в sentence-transformer.

Источники (в порядке применения):
  1. Ручной словарь (MANUAL_EXPANSIONS)  - высокоточные, приоритет
  2. Wikipedia API (ru → en, параллельно) - бесплатно, без ключа
  3. Wikidata API (параллельно)           - официальные синонимы/aliases
  4. Gemini API                           - батч-генерация для остальных
  5. Fallback: оставляем оригинал

Результат: expansions.json - словарь {domain: "expanded_text"}
Поддерживает возобновление: уже обработанные домены не запрашиваются повторно.

Запуск:
  python3 build_expansions.py            # build всё
  python3 build_expansions.py --rebuild  # стереть кэш и пересобрать
  python3 build_expansions.py --no-gemini # только Wikipedia + Wikidata
  python3 build_expansions.py --stats    # только показать статистику
"""

import sys, os, json, re, time, unicodedata, argparse, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path(__file__).parent

# ──────────────────────────────────────────────────────────────
#  АРГУМЕНТЫ
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Строим expansions.json")
parser.add_argument("--rebuild",    action="store_true", help="Сбросить кэш и пересобрать")
parser.add_argument("--no-gemini",  action="store_true", help="Не использовать Gemini API")
parser.add_argument("--stats",      action="store_true", help="Показать статистику и выйти")
parser.add_argument("--batch-size", type=int, default=20, help="Размер батча Gemini (default: 20)")
parser.add_argument("--workers",    type=int, default=10, help="Параллельных потоков Wikipedia (default: 10)")
args = parser.parse_args()

W = 72
print("=" * W)
print("BUILD EXPANSIONS - Wikipedia + Wikidata + Gemini")
print("=" * W)

# ──────────────────────────────────────────────────────────────
#  НОРМАЛИЗАЦИЯ
# ──────────────────────────────────────────────────────────────
DASH_VARIANTS = {
    "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
}

def normalize_label(label: str) -> str:
    """Normalizes domain label text for stable cache keys.

    Input:
      - label: raw domain text.
    Output:
      - Normalized label string.
    """
    label = unicodedata.normalize("NFC", label)
    for ch, repl in DASH_VARIANTS.items():
        label = label.replace(ch, repl)
    return re.sub(r"\s+", " ", label.lower()).strip()


# ──────────────────────────────────────────────────────────────
#  РУЧНОЙ СЛОВАРЬ
# ──────────────────────────────────────────────────────────────
MANUAL_EXPANSIONS = {
    # ── Финансы ───────────────────────────────────────────────────
    "финтех":                   "финтех финансовые технологии fintech платёжные сервисы онлайн-финансы",
    "банкинг":                  "банкинг банковский сектор банковские услуги финансовые операции",
    "e-commerce":               "e-commerce электронная коммерция интернет-торговля онлайн-магазин маркетплейс",
    "интернет-торговля":        "интернет-торговля e-commerce электронная коммерция онлайн-магазин",
    "электронная коммерция":    "электронная коммерция e-commerce интернет-торговля онлайн-ритейл",
    "adtech":                   "adtech рекламные технологии programmatic advertising dsp ssp rtb",
    "рекламные технологии":     "рекламные технологии adtech programmatic dsp ssp rtb реклама в интернете",
    "медицина":                 "медицина здравоохранение медицинские информационные системы клиника",
    "логистика":                "логистика управление цепочками поставок склад доставка транспорт",
    "туризм":                   "туризм travel-tech онлайн-бронирование путешествия авиабилеты отели",
    "crm":                      "crm управление взаимоотношениями с клиентами customer relationship management",
    "биллинг":                  "биллинг выставление счетов billing платёжные операции тарификация",
    "страхование":              "страхование страховые продукты insurance страховые выплаты полис",
    "ритейл":                   "ритейл розничная торговля retail продажи магазин",
    # ── QA / Тестирование ─────────────────────────────────────────
    "тестирование программного обеспечения":
                                "тестирование программного обеспечения software testing QA quality assurance тест тестировщик проверка ПО",
    "обеспечение качества программного обеспечения":
                                "обеспечение качества программного обеспечения QA software quality assurance тестирование контроль ПО",
    "автоматизация тестирования":
                                "автоматизация тестирования test automation QA автотесты CI/CD тестировщик selenium pytest",
    "управление дефектами":     "управление дефектами defect management bug tracking баг-трекер jira issues тестирование",
    "управление инцидентами":   "управление инцидентами incident management ITSM сбои аварии helpdesk support",
    "управление качеством релизов":
                                "управление качеством релизов release management QA gate релизный процесс deployment",
    "контроль качества электроники":
                                "контроль качества электроники electronics QA hardware testing verification",
    # ── Игры / Геймдев ────────────────────────────────────────────
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
}

# ──────────────────────────────────────────────────────────────
#  ЗАГРУЗКА ДОМЕНОВ
# ──────────────────────────────────────────────────────────────
candidates = {int(k): v for k, v in
              json.loads((BASE / "candidates_normalized.json").read_text()).items()}
ALL_DOMAINS = sorted({normalize_label(d) for v in candidates.values() for d in v})
print(f"  Уникальных доменов: {len(ALL_DOMAINS)}")
print(f"  Ручной словарь:     {len(MANUAL_EXPANSIONS)} записей")

# ──────────────────────────────────────────────────────────────
#  КЭШ
# ──────────────────────────────────────────────────────────────
CACHE_FILE = BASE / "expansions.json"

if args.rebuild and CACHE_FILE.exists():
    CACHE_FILE.unlink()
    print("  [INFO] Кэш сброшен (--rebuild)")

cache: dict[str, str] = {}
if CACHE_FILE.exists():
    cache = json.loads(CACHE_FILE.read_text())
    print(f"  [INFO] Загружен кэш: {len(cache)} записей")

# Ручные expansions всегда приоритетны
for d, exp in MANUAL_EXPANSIONS.items():
    cache[d] = exp

def save_cache():
    """Persists current expansion cache to JSON file.

    Input:
      - Uses global `CACHE` and output path constants.
    Output:
      - None. Writes cache file to disk.
    """
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


# ──────────────────────────────────────────────────────────────
#  СТАТИСТИКА
# ──────────────────────────────────────────────────────────────
def print_stats():
    """Prints statistics of cached expansion sources and coverage.

    Input:
      - Uses global cache and source counters.
    Output:
      - None. Writes statistics to stdout.
    """
    covered = sum(1 for d in ALL_DOMAINS if d in cache and cache[d] != d)
    manual_c, wiki_c, llm_c, fallback_c = 0, 0, 0, 0
    for d in ALL_DOMAINS:
        exp = cache.get(d, d)
        if d in MANUAL_EXPANSIONS:
            manual_c += 1
        elif "[src:wiki]" in exp:
            wiki_c += 1
        elif "[src:gemini]" in exp:
            llm_c += 1
        else:
            fallback_c += 1
    pct = covered * 100 // max(len(ALL_DOMAINS), 1)
    print(f"\n  Статистика expansions:")
    print(f"    Всего доменов:       {len(ALL_DOMAINS)}")
    print(f"    Покрыто:             {covered} ({pct}%)")
    print(f"      - ручной словарь:  {manual_c}")
    print(f"      - Wikipedia/Data:  {wiki_c}")
    print(f"      - LLM:             {llm_c}")
    print(f"    Без расширения:      {fallback_c}")

if args.stats:
    print_stats()
    sys.exit(0)

# ──────────────────────────────────────────────────────────────
#  HTTP HELPER
# ──────────────────────────────────────────────────────────────
UA = "DomainExpansionBot/1.0"

def _http_get(url: str, timeout: int = 6) -> dict | None:
    """Performs HTTP GET and parses JSON response safely.

    Input:
      - url: request URL.
      - timeout: request timeout in seconds.
    Output:
      - Parsed dictionary on success, otherwise None.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
#  WIKIPEDIA API
# ──────────────────────────────────────────────────────────────
STOP_WORDS = {
    "в", "на", "с", "и", "или", "а", "но", "не", "по", "из", "за", "к", "у",
    "о", "об", "что", "это", "как", "для", "при", "от", "до", "со", "та",
    "the", "a", "an", "of", "in", "is", "are", "was", "and", "or", "for",
    "to", "that", "with", "by", "as", "at", "be", "it", "this", "from",
    "also", "used", "which", "type", "can", "has", "have", "more", "than",
}

def _extract_kw(text: str, max_kw: int = 25) -> str:
    """Extracts compact keyword sequence from source text.

    Input:
      - text: source text from wiki/wikidata/LLM.
      - max_kw: maximum number of keywords in output.
    Output:
      - Space-separated keyword string.
    """
    words = re.findall(r"[a-zа-яё][a-zа-яё\-]*", text.lower())
    seen, unique = set(), []
    for w in words:
        if len(w) > 3 and w not in STOP_WORDS and w not in seen:
            seen.add(w)
            unique.append(w)
    return " ".join(unique[:max_kw])


def fetch_wiki(domain: str) -> str:
    """
    Параллельно-безопасная функция: запрашивает Wikipedia,
    возвращает enriched строку или "".
    """
    def _try(term: str, lang: str) -> str:
        enc = urllib.parse.quote(term.replace(" ", "_"))
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{enc}"
        data = _http_get(url)
        if not data:
            return ""
        t = data.get("type", "")
        if "not_found" in t or "disambiguation" in t:
            return ""
        extract = data.get("extract", "")
        if len(extract) < 60:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", extract)
        short = " ".join(sentences[:3])[:300]
        return _extract_kw(short)

    # 1. RU Wikipedia
    kw = _try(domain, "ru")
    if kw:
        return f"{domain} {kw} [src:wiki]"

    # 2. Capitalize
    kw = _try(domain.capitalize(), "ru")
    if kw:
        return f"{domain} {kw} [src:wiki]"

    # 3. EN Wikipedia (для английских терминов)
    if re.search(r"[a-zA-Z]", domain):
        kw = _try(domain, "en")
        if kw:
            return f"{domain} {kw} [src:wiki]"

    return ""


# ──────────────────────────────────────────────────────────────
#  WIKIDATA API
# ──────────────────────────────────────────────────────────────
def fetch_wikidata(domain: str) -> str:
    """Получаем aliases из Wikidata (параллельно-безопасная)."""
    enc = urllib.parse.quote(domain)
    url = (f"https://www.wikidata.org/w/api.php?action=wbsearchentities"
           f"&search={enc}&language=ru&limit=3&format=json")
    data = _http_get(url)
    if not data:
        return ""

    synonyms = []
    for item in data.get("search", [])[:2]:
        label   = item.get("label", "").lower()
        aliases = item.get("aliases", [])
        desc    = item.get("description", "")
        # Проверяем что нашли именно наш домен (минимальная проверка)
        dom_words = set(domain.lower().split()[:2])
        if not any(w in label for w in dom_words):
            continue
        synonyms.extend([a.lower() for a in aliases[:4]])
        if desc:
            synonyms.extend(re.findall(r"[a-zа-яё][a-zа-яё\-]{3,}", desc.lower())[:4])

    # Для EN-терминов пробуем EN Wikidata
    if not synonyms and re.search(r"[a-zA-Z]", domain):
        url_en = (f"https://www.wikidata.org/w/api.php?action=wbsearchentities"
                  f"&search={enc}&language=en&limit=2&format=json")
        data_en = _http_get(url_en)
        if data_en:
            for item in data_en.get("search", [])[:1]:
                synonyms.extend([a.lower() for a in item.get("aliases", [])[:3]])

    if not synonyms:
        return ""

    seen, unique = set(), []
    for s in synonyms:
        if s not in seen and s.lower() != domain.lower() and len(s) > 2:
            seen.add(s)
            unique.append(s)
    return " ".join(unique[:6])


def fetch_domain_info(domain: str) -> tuple[str, str]:
    """Комбинирует Wikipedia + Wikidata. Возвращает (domain, expansion)."""
    wiki = fetch_wiki(domain)
    wdata = fetch_wikidata(domain)

    if wiki and wdata:
        combined = f"{wiki} {wdata}"
        return domain, re.sub(r"\s+", " ", combined).strip()
    elif wiki:
        return domain, wiki
    elif wdata:
        return domain, f"{domain} {wdata} [src:wiki]"
    else:
        return domain, ""


# ──────────────────────────────────────────────────────────────
#  LLM - через llm_client.py (Gemini / OpenAI-совместимые)
# ──────────────────────────────────────────────────────────────
import llm_client

LLM_AVAILABLE = llm_client.is_available()

print(f"  LLM провайдер:      {llm_client.provider_info()}")
print(f"  LLM API-токен:      {'✓ задан' if LLM_AVAILABLE else '✗ не задан (LLM отключён)'}")

BATCH_PAUSE = 5

EXPANSION_PROMPT_TEMPLATE = """Ты - эксперт по IT и бизнес-доменам.

Для каждого бизнес-домена из списка сгенерируй расширение:
  - синонимы на русском языке (3-5 слов/фраз)
  - синонимы или термины на английском языке (3-5 слов/фраз)
  - смежные понятия (2-3)

Правила:
- Лаконично: каждый элемент - 1-4 слова
- Для IT-терминов (Kafka, Redis и т.п.) - укажи категорию ("брокер сообщений")
- Если домен непонятен - верни только оригинал в "ru"
- Отвечай ТОЛЬКО валидным JSON-массивом, без markdown, без пояснений

Формат:
[{{"domain": "...", "ru": ["...", "..."], "en": ["...", "..."], "related": ["..."]}}]

Домены:
{domains_list}"""


def llm_expand_batch(domains: list[str]) -> dict[str, str]:
    """Расширяет батч доменов через LLM-провайдер из llm_client."""
    prompt = EXPANSION_PROMPT_TEMPLATE.format(domains_list="\n".join(domains))
    try:
        raw = llm_client.call(prompt)
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$",     "", raw)

        parsed = json.loads(raw)
        result = {}
        for item in parsed:
            d = normalize_label(item.get("domain", ""))
            if not d:
                continue
            parts = (
                [d]
                + item.get("ru", [])
                + item.get("en", [])
                + item.get("related", [])
                + ["[src:gemini]"]   # тег сохраняем для обратной совместимости
            )
            result[d] = re.sub(r"\s+", " ", " ".join(parts)).strip()
        return result

    except json.JSONDecodeError as e:
        print(f"    [ERR] JSON parse: {e}")
        return {}
    except RuntimeError as e:
        print(f"    [ERR] LLM: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
#  ОСНОВНОЙ ЦИКЛ
# ──────────────────────────────────────────────────────────────
remaining = [d for d in ALL_DOMAINS if d not in cache or cache[d] == d]
print(f"\n  К обработке: {len(remaining)} доменов")
print(f"  Уже в кэше:  {len(ALL_DOMAINS) - len(remaining)} доменов")

# ── Шаг 1: Wikipedia + Wikidata (параллельно) ────────────────
print()
print("─" * W)
print(f"ШАГ 1 - Wikipedia + Wikidata  ({args.workers} потоков)")
print("─" * W)

t0 = time.time()
wiki_hits = 0
still_empty: list[str] = []

with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(fetch_domain_info, d): d for d in remaining}
    done = 0
    for future in as_completed(futures):
        domain, expansion = future.result()
        done += 1
        if expansion:
            cache[domain] = expansion
            wiki_hits += 1
        else:
            still_empty.append(domain)

        if done % 50 == 0 or done == len(remaining):
            pct = done * 100 // len(remaining)
            elapsed = time.time() - t0
            print(f"  [{done:3d}/{len(remaining)}] {pct}%  "
                  f"хиты={wiki_hits}  без_ответа={len(still_empty)}  {elapsed:.0f}с", flush=True)

# Сохраняем после Wikipedia
save_cache()
elapsed = time.time() - t0
print(f"\n  Wikipedia/Wikidata: {wiki_hits} хитов из {len(remaining)} за {elapsed:.1f}с")
print(f"  Осталось без расширения: {len(still_empty)}")

# ── Шаг 2: LLM для оставшихся ────────────────────────────────
if still_empty and not args.no_gemini and LLM_AVAILABLE:
    print()
    print("─" * W)
    print(f"ШАГ 2 - LLM ({llm_client.provider_info()}, батч={args.batch_size}, домены={len(still_empty)})")
    print("─" * W)

    batches = [still_empty[i:i + args.batch_size]
               for i in range(0, len(still_empty), args.batch_size)]
    print(f"  Батчей: {len(batches)}")

    llm_hits = 0
    for bi, batch in enumerate(batches, 1):
        print(f"\n  [BATCH {bi}/{len(batches)}] {len(batch)} доменов")
        result = llm_expand_batch(batch)
        for d, exp in result.items():
            if d in ALL_DOMAINS:
                cache[d] = exp
                llm_hits += 1

        save_cache()
        print(f"    → Получено: {len(result)}  (итого LLM: {llm_hits})")

        if bi < len(batches):
            print(f"    [INFO] Пауза {BATCH_PAUSE}с...")
            time.sleep(BATCH_PAUSE)

elif still_empty and args.no_gemini:
    print(f"\n  [INFO] --no-gemini: пропускаем {len(still_empty)} доменов")
elif still_empty and not LLM_AVAILABLE:
    print(f"\n  [INFO] LLM не настроен. {len(still_empty)} доменов без расширения.")
    print(f"         Задайте api_key в config.json (секция 'llm') или LLM_API_KEY=...")

# ── Итог ─────────────────────────────────────────────────────
print()
print("=" * W)
print("ГОТОВО")
save_cache()
print_stats()
print()
print(f"  Файл: {CACHE_FILE}")

print()
print("  Примеры расширений:")
examples = [
    "финтех", "apache kafka", "телемедицина", "foodtech",
    "kyc-процессы", "эквайринг", "igaming", "hr-технологии",
]
for d in examples:
    exp = cache.get(d, "(нет в кэше)")
    tag = "manual" if d in MANUAL_EXPANSIONS else ("gemini" if "[src:gemini]" in exp else ("wiki" if "[src:wiki]" in exp else "-"))
    display = re.sub(r"\[src:\w+\]", "", exp).strip()[:88]
    print(f"  [{tag:6s}] {d:<35s} → {display}")
