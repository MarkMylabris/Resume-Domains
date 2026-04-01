"""
parse_hh_resumes.py - Парсинг и доменная индексация резюме из HH.ru датасета.

Пайплайн:
  1. Читает CSV (dst-3.0_16_1_hh_database.csv)
  2. Фильтрует IT-профили по ключевым словам в должности/опыте
  3. Балансирует выборку: лимит N резюме на каждую должность (--max-per-role)
  4. Очищает поле «Опыт работы»: убирает даты, URL, HR-шаблоны, зарплаты
  5. Извлекает 5–8 бизнес-доменов из очищенного текста (семантический поиск + MMR)
  6. Сохраняет результат в datasets/hh_ru_resumes/candidates_hh_full.json

Рекомендованный запуск (полный сбалансированный датасет ~23k IT-резюме, ~10 мин):
    python parse_hh_resumes.py --all --it-only --max-per-role 50

Другие варианты:
    python parse_hh_resumes.py                             # первые 200 резюме (быстрый тест)
    python parse_hh_resumes.py --limit 2000 --it-only      # 2000 IT-профилей
    python parse_hh_resumes.py --show 5                    # показать 5 примеров без сохранения
    python parse_hh_resumes.py --all                       # все ~44k (без фильтрации, ~30 мин)
"""

import sys, os, re, csv, json, argparse, unicodedata
import numpy as np
from pathlib import Path
from collections import Counter

os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE    = Path(__file__).parent
DATASET = BASE / "datasets" / "hh_ru_resumes" / "dst-3.0_16_1_hh_database.csv"
OUT     = BASE / "datasets" / "hh_ru_resumes" / "candidates_hh.json"

# ──────────────────────────────────────────────────────────────
#  ОЧИСТКА ТЕКСТА РЕЗЮМЕ
# ──────────────────────────────────────────────────────────────

# Месяцы для удаления дат типа "Июнь 2015 - по настоящее время 3 года 11 месяцев"
_MONTHS = (
    r"январ\w*|феврал\w*|март\w*|апрел\w*|май\w*|июн\w*|июл\w*|"
    r"август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*"
)

# Паттерны шума в тексте резюме HH.ru
_NOISE_PATTERNS = [
    # Периоды: "Июнь 2015 - по настоящее время 3 года 11 месяцев"
    (re.compile(rf"({_MONTHS})\s+\d{{4}}\s*[-–\-]+\s*(по настоящее время|\d{{4}})",
                re.IGNORECASE), " "),
    # Стаж: "Опыт работы 22 года 9 месяцев", "3 года 11 месяцев"
    (re.compile(r"\d+\s*(год|лет|месяц)\w*\s+\d*\s*(год|лет|месяц)?\w*",
                re.IGNORECASE), " "),
    # Одиночные годы: "2015", "2019" - убираем только вне слова (т.е. не "Python3")
    (re.compile(r"(?<!\w)(19|20)\d{2}(?!\w)"), " "),
    # URL, email, домены
    (re.compile(r"https?://\S+|www\.\S+|\S+\.(ru|com|org|net|by|ua|kz)\S*",
                re.IGNORECASE), " "),
    (re.compile(r"\S+@\S+\.\S+"), " "),
    # HR-шаблоны HH.ru (колонки попадают в поле "Опыт работы")
    (re.compile(r"(Занятость|График\s*работы|Опыт работы|График|Удалённая работа)"
                r"\s*:\s*[^\n.]*", re.IGNORECASE), " "),
    # "полная занятость", "гибкий график" как отдельные фразы
    (re.compile(r"\b(полная|частичная|проектная)\s+занятость\b", re.IGNORECASE), " "),
    (re.compile(r"\b(гибкий|полный|сменный|вахтовый)\s+день\b", re.IGNORECASE), " "),
    (re.compile(r"\bудалённая\s+работа\b", re.IGNORECASE), " "),
    # Тип компании из HH: "Информационные технологии, системная интеграция, интернет ..."
    (re.compile(r"Информационные технологии,\s*системная интеграция,\s*интернет\s*\.{0,3}",
                re.IGNORECASE), " "),
    # Зарплата: "250 000 руб.", "60 000 бел. руб."
    (re.compile(r"\d[\d\s]*\d\s*(руб|бел\.\s*руб|usd|eur)\s*\.?", re.IGNORECASE), " "),
    # Юридические формы: "ООО", "ПАО", "ЗАО", "АО" в начале названий
    (re.compile(r"\b(ООО|ПАО|ЗАО|АО|ОАО|ИП|ГУП|МУП|НКО)\b\s*[«\"']?"), " "),
    # Лишние знаки препинания и спецсимволы
    (re.compile(r'[»«„""\*\|#@^~`]'), " "),
    # Повторяющиеся пробелы и тире
    (re.compile(r"[\s\-–-]{3,}"), " "),
]


def clean_resume(text: str) -> str:
    """Убирает шаблонный мусор, сохраняя смысловые фрагменты."""
    for pattern, repl in _NOISE_PATTERNS:
        text = pattern.sub(repl, text)
    # Схлопываем пробелы
    text = re.sub(r"[ \t]+", " ", text)
    # Убираем строки короче 3 символов (остатки после очистки)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 3]
    return " ".join(lines).strip()


# ──────────────────────────────────────────────────────────────
#  ИЗВЛЕЧЕНИЕ ДОМЕНОВ (MMR - как в live_demo.py)
# ──────────────────────────────────────────────────────────────

import scorer  # загружаем после env-переменных

# Домены, которые слишком семантически универсальны и попадают в большинство IT-резюме.
# Они корректны в контексте scoring (узкие кандидаты), но не подходят для ИЗВЛЕЧЕНИЯ
# из произвольных текстов. Исключаем их при парсинге резюме.
_DOMAIN_BLOCKLIST = {
    # ── Слишком универсальные IT/management термины ──────────────────────────
    # присутствуют в 15-20% резюме, не несут специфики
    "управление персоналом",                          # HR у всех тимлидов
    "управление задачами",                            # >21% резюме - шум
    "ит-консалтинг",                                  # 18% - слишком широко
    "разработка программного обеспечения",            # 17% - generic
    "системная интеграция",                           # 17% - generic
    "электронный документооборот",                    # 17% - EDO есть везде
    "интернет-компании",                              # 17% - не домен
    "управление клиентской базой",                    # 17% - generic CRM
    "управление проектами",                           # 15% - generic PM
    "корпоративные информационные системы",           # 12% - generic
    "система управления взаимоотношениями с клиентами", # 11% - generic CRM
    "управление продавцами",                          # 11% - generic sales
    "веб-сервис транспортных сотрудников",            # расширение даёт широкий охват
    "интернет-компания",                              # дубль "интернет-компании"
    "интеграция систем",                              # generic
    # ── Отраслевые «ложные совпадения» ──────────────────────────────────────
    "управление проектами в машиностроении",          # ложный PM-матч для IT
    "управление строительными объектами",             # аналогично
    # ── Артефакты HH-формата и расширений ───────────────────────────────────
    "mes система",                                    # MES ≈ любое IT-управление
    "цифровой ассистент госуслуг",                    # "цифровой"+"ассистент" → generic
    "конструктор административных панелей",           # admin panel builder
    "панель администратора",                          # dashboard → широкое
    "автоматизация бизнес-процессов государственных услуг",  # слишком специфично
    "управление корзиной",                            # e-commerce частность
    "интерактивные доски в онлайн-образовании",       # ложное срабатывание
    "проведение ido",                                 # IDO/ICO crypto - ложное
    "проведение a/b-экспериментов",                   # а/б-тест - широкое
    "ритейл электроники",                             # без контекста ложное
    "веб-браузер",                                    # не бизнес-домен
    "цифровая трансформация бизнес-процессов",        # слишком общее
    "личный кабинет государственных услуг",           # госуслуги → ложное
    "государственные информационные услуги",          # аналогично
    "корпоративный софт для автоматизации",           # 6.6% - generic
    "розничная сеть электроники и бытовой техники",   # 4.4% - слишком специфично
}


def extract_domains(text: str, top_k: int = 9, mmr_lambda: float = 0.65,
                    min_sim: float = 0.50) -> list[str]:
    """Извлекает top_k разнообразных бизнес-доменов из текста резюме.

    min_sim - минимальный порог косинусного сходства (основной). Если ни один
    домен его не достигает - fallback к более мягкому порогу 0.44, чтобы хоть
    что-то вернуть для нетипичных или коротких текстов.
    top_k=9 при min_sim=0.50 даёт целевое среднее 4-5 доменов/резюме.
    """
    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        text_emb = scorer._model.encode([text], normalize_embeddings=True)[0]

    sims = scorer._emb_v7 @ text_emb

    # Индексы заблокированных доменов
    blocked = {i for i, d in enumerate(scorer._domain_list) if d in _DOMAIN_BLOCKLIST}

    def _mmr_select(min_threshold: float) -> list[int]:
        candidates = [
            i for i in np.argsort(sims)[::-1][:200]
            if float(sims[i]) >= min_threshold and i not in blocked
        ]
        selected, selected_embs = [], []
        for _ in range(top_k):
            best_score, best_i = -1.0, -1
            for ci in candidates:
                if ci in selected:
                    continue
                rel = float(sims[ci])
                red = max((float(scorer._emb_v7[ci] @ e) for e in selected_embs), default=0.0)
                score = mmr_lambda * rel - (1 - mmr_lambda) * red
                if score > best_score:
                    best_score, best_i = score, ci
            if best_i == -1:
                break
            selected.append(best_i)
            selected_embs.append(scorer._emb_v7[best_i])
        return selected

    result = _mmr_select(min_sim)
    if not result:                          # fallback: ничего не нашлось → снижаем порог
        result = _mmr_select(0.44)

    return [scorer._domain_list[i] for i in result]


# ──────────────────────────────────────────────────────────────
#  ЗАГРУЗКА CSV
# ──────────────────────────────────────────────────────────────

IT_KEYWORDS = [
    "разработ", "программист", "python", "java", "backend", "frontend",
    "devops", "data", "аналитик", "тестировщ", "qa", "архитектор",
    "инженер", "системный", "финтех", "product", "менеджер проект",
    "software", "developer", "golang", "c++", "php", ".net", "android",
    "ios", "mobile", "cloud", "kubernetes", "docker", "security", "ux", "ui",
]


def is_it_profile(row: list[str]) -> bool:
    """Checks whether a HH CSV row belongs to IT-related profile.

    Input:
      - row: CSV row list as parsed from dataset file.
    Output:
      - True if row matches IT keywords, otherwise False.
    """
    text = " ".join(row[i] for i in (2, 6, 8) if i < len(row)).lower()
    return any(kw in text for kw in IT_KEYWORDS)


def load_csv(path: Path, limit: int | None, it_only: bool,
             max_per_role: int | None = None, seed: int = 42) -> list[dict]:
    """Loads HH CSV rows and applies IT filter and optional role balancing.

    Input:
      - path: source CSV file path.
      - limit: max rows to return, None for full scan.
      - it_only: if True, keeps only rows matched by IT keyword filter.
      - max_per_role: optional cap per normalized job title.
      - seed: random seed for deterministic sampling when limit is set.
    Output:
      - List of normalized row dictionaries for downstream parsing.
    """
    from collections import Counter as _Counter
    role_counts: _Counter = _Counter()
    rows = []

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) < 9:
                continue
            if it_only and not is_it_profile(row):
                continue
            if max_per_role:
                # Нормализуем должность до 50 симв. как ключ балансировки
                role_key = re.sub(r"\s+", " ", row[2].strip().lower())[:50]
                if role_counts[role_key] >= max_per_role:
                    continue
                role_counts[role_key] += 1
            rows.append({
                "position":    row[2].strip(),
                "last_role":   row[8].strip(),
                "exp_raw":     row[6].strip(),
                "education":   row[9].strip() if len(row) > 9 else "",
                "city":        row[3].split(",")[0].strip() if row[3] else "",
            })

    if limit and limit < len(rows):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=limit, replace=False)
        rows = [rows[i] for i in sorted(idx)]

    return rows


# ──────────────────────────────────────────────────────────────
#  СБОРКА «СИНТЕТИЧЕСКОГО» ТЕКСТА ДЛЯ КОДИРОВАНИЯ
# ──────────────────────────────────────────────────────────────

def build_input_text(row: dict) -> str:
    """Объединяет должность + очищенный опыт + образование в один текст."""
    parts = []
    # Должность повторяем дважды - она самый точный сигнал
    if row["position"]:
        parts.append(row["position"])
        parts.append(row["position"])
    if row["last_role"] and row["last_role"] != row["position"]:
        parts.append(row["last_role"])
    cleaned_exp = clean_resume(row["exp_raw"])
    if cleaned_exp:
        parts.append(cleaned_exp)
    # Образование - краткое, даёт контекст (напр. «Финансовый университет»)
    if row["education"]:
        edu_short = re.sub(r"\d{4}", "", row["education"])[:200].strip()
        parts.append(edu_short)
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────
#  ОСНОВНОЙ ПАЙПЛАЙН
# ──────────────────────────────────────────────────────────────

def run(rows: list[dict], show_n: int = 0, verbose: bool = False) -> dict:
    """Обрабатывает список строк CSV, возвращает словарь {id: [домены]}."""
    scorer._load()

    results  = {}
    counters = Counter()
    total    = len(rows)

    print(f"\n  Обработка {total} резюме...\n")

    for i, row in enumerate(rows, 1):
        input_text = build_input_text(row)
        if not input_text.strip():
            counters["empty"] += 1
            continue

        domains = extract_domains(input_text)
        results[i] = domains
        counters["ok"] += 1

        # Прогресс каждые 10 резюме
        if i % 10 == 0 or i == total:
            pct = i * 100 // total
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {i}/{total} ({pct}%)", end="", flush=True)

        # Детальный вывод первых show_n
        if show_n and i <= show_n:
            print(f"\n\n  ─── Резюме #{i} ───")
            print(f"  Должность:  {row['position'][:70]}")
            print(f"  Роль:       {row['last_role'][:70]}")
            exp_clean = clean_resume(row["exp_raw"])
            print(f"  Опыт (сырой, 300 симв):   {row['exp_raw'][:300]}")
            print(f"  Опыт (чистый, 300 симв):  {exp_clean[:300]}")
            print(f"  Извлечённые домены: {domains}")
            if verbose:
                print(f"  Входной текст для модели: {input_text[:400]}")

    print()  # перенос после прогресс-бара

    return results


def print_summary(results: dict, rows: list[dict]) -> None:
    """Prints parsing statistics for generated candidate domain dataset.

    Input:
      - results: mapping candidate_id -> extracted domain list.
      - rows: original loaded rows used for processing.
    Output:
      - None. Writes summary metrics to stdout.
    """
    print(f"\n  {'─'*60}")
    print(f"  Итого обработано:   {len(results)} резюме")
    all_domains = [d for doms in results.values() for d in doms]
    top_domains = Counter(all_domains).most_common(15)
    avg_domains = len(all_domains) / max(len(results), 1)
    print(f"  Среднее доменов/резюме: {avg_domains:.1f}")
    print(f"\n  Топ-15 извлечённых доменов:")
    for dom, cnt in top_domains:
        bar = "█" * min(cnt * 30 // max(top_domains[0][1], 1), 30)
        print(f"    {bar:<30}  {cnt:4d}  {dom}")


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for HH resume parsing pipeline.

    Input:
      - Command-line arguments for dataset filters, limits, preview mode, and output path.
    Output:
      - None. Prints progress/statistics and optionally saves JSON output.
    """
    parser = argparse.ArgumentParser(
        description="Парсинг резюме HH.ru и извлечение бизнес-доменов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python parse_hh_resumes.py                       # 200 резюме (тест)
  python parse_hh_resumes.py --limit 1000          # 1000 случайных
  python parse_hh_resumes.py --limit 500 --it-only # 500 IT-профилей
  python parse_hh_resumes.py --show 3              # показать 3 примера
  python parse_hh_resumes.py --all                 # все ~44k резюме
""",
    )
    parser.add_argument("--limit",        type=int, default=200, help="Сколько резюме взять (default: 200)")
    parser.add_argument("--all",          action="store_true",   help="Обработать весь датасет")
    parser.add_argument("--it-only",      action="store_true",   help="Только IT-профили")
    parser.add_argument("--max-per-role", type=int, default=None,help="Лимит резюме на должность (балансировка)")
    parser.add_argument("--show",         type=int, default=0,   help="Показать N детальных примеров")
    parser.add_argument("--verbose",      action="store_true",   help="Показывать полный входной текст")
    parser.add_argument("--no-save",      action="store_true",   help="Не сохранять результат")
    parser.add_argument("--out",          type=str, default=str(OUT), help="Путь для сохранения JSON")
    args = parser.parse_args()

    if not DATASET.exists():
        print(f"\n  ✗ Датасет не найден: {DATASET}")
        print(f"  Скачайте: https://drive.google.com/file/d/1Kb78mAWYKcYlellTGhIjPI-bCcKbGuTn")
        sys.exit(1)

    print(f"\n  Загрузка датасета: {DATASET.name}...")
    limit = None if args.all else args.limit
    rows  = load_csv(DATASET, limit=limit, it_only=args.it_only,
                     max_per_role=args.max_per_role)
    info  = []
    if args.it_only:      info.append("только IT")
    if args.max_per_role: info.append(f"лимит {args.max_per_role}/роль")
    print(f"  Загружено строк: {len(rows)}" + (f" ({', '.join(info)})" if info else ""))

    print(f"\n  Инициализация модели...", end=" ", flush=True)
    scorer._load()
    print("готово.")

    results = run(rows, show_n=args.show, verbose=args.verbose)
    print_summary(results, rows)

    if not args.no_save and not args.show:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\n  ✓ Сохранено: {out_path}  ({len(results)} записей)")
    elif args.show:
        print(f"\n  (--show режим: сохранение отключено)")


if __name__ == "__main__":
    main()
