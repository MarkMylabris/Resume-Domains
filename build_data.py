"""
build_data.py - Подготовка всех данных с нуля.

Шаги:
  1. parse       - парсинг SQL -> кандидаты с доменами
  2. normalize   - нормализация меток + co-occurrence матрица
  3. embed       - эмбеддинги доменов (v7: авто-расширения)
  4. taxonomy    - L1/L2 таксономия через LLM API (опционально)
  5. cooccur-hh  - пересборка co-occurrence из HH.ru датасета (~24k резюме)

Зависимости:
  Шаги 1-4 последовательны. Каждый шаг кэширует результат -
  повторный запуск пропускает уже выполненные шаги.
  Шаг cooccur-hh независим - требует только candidates_hh_full.json.

Запуск:
    python build_data.py resumes_domains_rows.sql         # парсинг + нормализация + эмбеддинги
    python build_data.py resumes_domains_rows.sql --all   # все шаги
    python build_data.py resumes_domains_rows.sql --step embed  # конкретный шаг
    python build_data.py resumes_domains_rows.sql --step cooccur-hh  # только co-occurrence из HH
    python build_data.py resumes_domains_rows.sql --rebuild     # принудительно пересобрать

Требования к SQL-файлу:
    VALUES ('id', '"{\"домен1\",\"домен2\"}\"')
"""

import sys, os, re, json, math, unicodedata, warnings, argparse, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"]         = "1"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / ".venv_libs"))

import io, contextlib
_buf = io.StringIO()


# ──────────────────────────────────────────────────────────────
#  УТИЛИТЫ
# ──────────────────────────────────────────────────────────────
_DASH_MAP = {"\u2011": "-", "\u2012": "-", "\u2013": "-",
             "\u2014": "-", "\u2015": "-", "\u2212": "-"}


def _norm(label: str) -> str:
    """Нормализует метку домена."""
    label = unicodedata.normalize("NFC", label)
    for ch, repl in _DASH_MAP.items():
        label = label.replace(ch, repl)
    return re.sub(r"\s+", " ", label.lower()).strip()


def _log(msg: str) -> None:
    """Prints standardized pipeline log line.

    Input:
      - msg: text message to print.
    Output:
      - None. Writes formatted log line to stdout.
    """
    print(f"  {msg}")


# ──────────────────────────────────────────────────────────────
#  ШАГ 1: ПАРСИНГ SQL
# ──────────────────────────────────────────────────────────────
def step_parse(sql_file: Path) -> dict[int, list[str]]:
    """Читает SQL-файл, извлекает кандидатов и их домены."""
    _log(f"Парсинг {sql_file.name} ...")
    raw = sql_file.read_text(encoding="utf-8")

    row_pattern    = re.compile(r"\('(\d+)',\s*'(.*?)'\)", re.DOTALL)
    domain_pattern = re.compile(r'\\"([^\\"]+)\\"')

    candidates: dict[int, list[str]] = {}
    errors = []

    for cid_str, value_raw in row_pattern.findall(raw):
        cid     = int(cid_str)
        domains = domain_pattern.findall(value_raw)
        if domains:
            candidates[cid] = domains
        else:
            errors.append(cid)

    _log(f"Кандидатов: {len(candidates)}, ошибок парсинга: {len(errors)}")
    if errors:
        _log(f"  [!] Без доменов: {errors[:10]}")

    all_labels = [d for v in candidates.values() for d in v]
    _log(f"Меток всего: {len(all_labels)}, уникальных: {len(set(all_labels))}")
    return candidates


# ──────────────────────────────────────────────────────────────
#  ШАГ 2: НОРМАЛИЗАЦИЯ + CO-OCCURRENCE
# ──────────────────────────────────────────────────────────────
def step_normalize(candidates_raw: dict[int, list[str]]) -> tuple[dict, dict]:
    """Нормализует метки и строит матрицу совместных вхождений."""
    _log("Нормализация меток ...")

    candidates: dict[int, list[str]] = {}
    changes, dupes = 0, 0

    for cid, domains in candidates_raw.items():
        normed, seen = [], set()
        for d in domains:
            nd = _norm(d)
            if nd != d:
                changes += 1
            if nd in seen:
                dupes += 1
            else:
                seen.add(nd)
                normed.append(nd)
        candidates[cid] = normed

    total_after = sum(len(v) for v in candidates.values())
    unique_after = len(set(d for v in candidates.values() for d in v))
    _log(f"Изменений нормализации: {changes}, дублей удалено: {dupes}")
    _log(f"Итого меток: {total_after}, уникальных доменов: {unique_after}")

    # Co-occurrence
    _log("Строим co-occurrence матрицу ...")
    cooccur: dict[str, Counter] = defaultdict(Counter)
    for domains in candidates.values():
        for a, b in combinations(list(dict.fromkeys(domains)), 2):
            cooccur[a][b] += 1
            cooccur[b][a] += 1

    pairs = sum(len(v) for v in cooccur.values()) // 2
    _log(f"Уникальных пар с co-occurrence > 0: {pairs}")

    return candidates, {k: dict(v.most_common(20)) for k, v in cooccur.items()}


# ──────────────────────────────────────────────────────────────
#  ШАГ 3: ЭМБЕДДИНГИ v7
# ──────────────────────────────────────────────────────────────
def step_embed(candidates: dict[int, list[str]], rebuild: bool = False) -> None:
    """Строит эмбеддинги доменов с расширениями из expansions.json."""
    from sentence_transformers import SentenceTransformer

    out_file = BASE / "domain_embeddings_v7.npz"
    if out_file.exists() and not rebuild:
        _log(f"Эмбеддинги уже есть ({out_file.name}) - пропускаем (--rebuild для пересборки)")
        return

    # Загружаем расширения
    expansions = {}
    exp_file = BASE / "expansions.json"
    if exp_file.exists():
        raw = json.loads(exp_file.read_text())
        for k, v in raw.items():
            expansions[_norm(k)] = re.sub(r"\[src:\w+\]", "", v).strip()
        _log(f"Расширений загружено из expansions.json: {len(expansions)}")
    else:
        _log("[!] expansions.json не найден - запустите build_expansions.py")

    domain_list = sorted(set(d for v in candidates.values() for d in v))
    texts       = [expansions.get(d, d) for d in domain_list]
    n_expanded  = sum(1 for d, t in zip(domain_list, texts) if t != d)
    _log(f"Доменов: {len(domain_list)}, расширено: {n_expanded} ({n_expanded*100//len(domain_list)}%)")

    _log("Загружаем модель SentenceTransformer ...")
    with contextlib.redirect_stderr(_buf):
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder=str(BASE / ".model_cache"),
        )

    _log("Кодируем домены ...")
    t0 = time.time()
    with contextlib.redirect_stderr(_buf):
        embs = model.encode(texts, normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False)
    _log(f"Закодировано за {time.time()-t0:.1f}с, shape={embs.shape}")

    np.savez_compressed(out_file,
                        domains=np.array(domain_list, dtype=object),
                        embeddings=embs)
    _log(f"Сохранено → {out_file.name}")


# ──────────────────────────────────────────────────────────────
#  ШАГ 4: ТАКСОНОМИЯ (Gemini)
# ──────────────────────────────────────────────────────────────
def step_taxonomy(candidates: dict[int, list[str]]) -> None:
    """Строит L1/L2 таксономию через Gemini API.
    Требует config.json с ключом gemini.api_key или переменную GEMINI_API_KEY.
    """
    out_file = BASE / "taxonomy.json"
    if out_file.exists():
        tax = json.loads(out_file.read_text())
        _log(f"Таксономия уже есть ({len(tax)} доменов) - пропускаем")
        return

    config = json.loads((BASE / "config.json").read_text()) \
             if (BASE / "config.json").exists() else {}
    api_key = (os.environ.get("GEMINI_API_KEY", "")
               or config.get("gemini", {}).get("api_key", ""))

    if not api_key or api_key.startswith("<"):
        _log("[!] GEMINI_API_KEY не задан - таксономия пропускается")
        _log("  Задайте ключ в config.json или через GEMINI_API_KEY=... python build_data.py ...")
        return

    # Если ключ есть - запускаем скрипт таксономии через subprocess (он уже реализован)
    _log("Запуск build_taxonomy.py (develop/improve4_gemini_taxonomy.py) ...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(BASE / "dev" / "improve4_gemini_taxonomy.py")],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        _log("Таксономия построена.")
    else:
        _log(f"[!] Ошибка при построении таксономии:\n{result.stderr[-500:]}")


# ──────────────────────────────────────────────────────────────
#  ШАГ 5: CO-OCCURRENCE ИЗ HH-ДАТАСЕТА
# ──────────────────────────────────────────────────────────────
def step_cooccur_hh(min_domains: int = 3, top_per_domain: int = 30) -> None:
    """Пересобирает cooccur_matrix.json из реального HH.ru датасета.

    Использует candidates_hh_full.json (~24k резюме) вместо 117 тестовых.
    Фильтрует пары с count < 2, оставляет top_per_domain самых частых соседей.
    """
    hh_path = BASE / "datasets" / "hh_ru_resumes" / "candidates_hh_full.json"
    if not hh_path.exists():
        _log("[!] candidates_hh_full.json не найден - запустите parse_hh_resumes.py")
        return

    raw = json.loads(hh_path.read_text())
    candidates = [v for v in raw.values() if len(v) >= min_domains]
    _log(f"HH-резюме для co-occurrence: {len(candidates)}")

    cooccur: dict[str, Counter] = defaultdict(Counter)
    for domains in candidates:
        uniq = list(dict.fromkeys(domains))      # сохраняем порядок, убираем дубли
        for a, b in combinations(uniq, 2):
            cooccur[a][b] += 1
            cooccur[b][a] += 1

    # Фильтруем редкие пары (< 2 вхождений) и сохраняем топ соседей
    filtered = {}
    for domain, neighbors in cooccur.items():
        top = {d: c for d, c in neighbors.most_common(top_per_domain) if c >= 2}
        if top:
            filtered[domain] = top

    pairs = sum(len(v) for v in filtered.values()) // 2
    _log(f"Уникальных пар co-occurrence (count >= 2): {pairs}")
    _log(f"Доменов в матрице: {len(filtered)}")

    out = BASE / "cooccur_matrix.json"
    out.write_text(json.dumps(filtered, ensure_ascii=False, indent=2))
    _log(f"Сохранено -> {out.name}")


# ──────────────────────────────────────────────────────────────
#  ТОЧКА ВХОДА
# ──────────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point for data build pipeline orchestration.

    Input:
      - Command-line arguments for SQL source, selected steps, and rebuild mode.
    Output:
      - None. Executes selected steps and writes generated artifacts to disk.
    """
    parser = argparse.ArgumentParser(
        description="Подготовка данных для скорера",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python build_data.py resumes_domains_rows.sql
  python build_data.py resumes_domains_rows.sql --all
  python build_data.py resumes_domains_rows.sql --step parse normalize
  python build_data.py resumes_domains_rows.sql --step embed --rebuild
  python build_data.py resumes_domains_rows.sql --step cooccur-hh
""",
    )
    parser.add_argument("sql_file",    help="Путь к SQL-файлу с резюме")
    parser.add_argument("--all",       action="store_true",
                        help="Все шаги включая таксономию (требует LLM API)")
    parser.add_argument("--step",      nargs="+",
                        choices=["parse", "normalize", "embed", "taxonomy", "cooccur-hh"],
                        help="Конкретные шаги (по умолчанию: parse normalize embed)")
    parser.add_argument("--rebuild",   action="store_true",
                        help="Принудительно пересобрать даже если кэш существует")
    args = parser.parse_args()

    sql_path = Path(args.sql_file)
    if not sql_path.exists():
        print(f"  [ERR] Файл не найден: {sql_path}")
        sys.exit(1)

    steps = args.step or ["parse", "normalize", "embed"]
    if args.all:
        steps = ["parse", "normalize", "embed", "taxonomy", "cooccur-hh"]

    print()
    print("┌─ BUILD DATA ─────────────────────────────────────────────────┐")
    print(f"│  SQL-файл: {sql_path.name:<52} │")
    print(f"│  Шаги:     {', '.join(steps):<52} │")
    print("└──────────────────────────────────────────────────────────────┘")
    print()

    candidates = None

    # ── Парсинг ──────────────────────────────────────────────
    cands_file = BASE / "candidates_normalized.json"
    if "parse" in steps or "normalize" in steps:
        if cands_file.exists() and not args.rebuild:
            _log(f"candidates_normalized.json уже есть - загружаем ...")
            candidates = {int(k): v for k, v in
                          json.loads(cands_file.read_text()).items()}
        else:
            print("─── Шаг 1: Парсинг SQL ───")
            raw_candidates = step_parse(sql_path)

            print("\n─── Шаг 2: Нормализация ───")
            candidates, cooccur = step_normalize(raw_candidates)

            # Сохраняем
            cands_file.write_text(
                json.dumps({str(k): v for k, v in candidates.items()},
                           ensure_ascii=False, indent=2))
            _log(f"Сохранено → {cands_file.name}")

            cooccur_file = BASE / "cooccur_matrix.json"
            cooccur_file.write_text(
                json.dumps(cooccur, ensure_ascii=False, indent=2))
            _log(f"Сохранено → {cooccur_file.name}")

    # ── Эмбеддинги ───────────────────────────────────────────
    if "embed" in steps:
        if candidates is None:
            if not cands_file.exists():
                _log("✗ Сначала запустите шаг parse (или --step parse normalize embed)")
                sys.exit(1)
            candidates = {int(k): v for k, v in
                          json.loads(cands_file.read_text()).items()}
        print("\n─── Шаг 3: Эмбеддинги v7 ───")
        step_embed(candidates, rebuild=args.rebuild)

    # ── Таксономия ───────────────────────────────────────────
    if "taxonomy" in steps:
        if candidates is None:
            candidates = {int(k): v for k, v in
                          json.loads(cands_file.read_text()).items()}
        print("\n─── Шаг 4: Таксономия (LLM) ───")
        step_taxonomy(candidates)

    # ── Co-occurrence из HH ──────────────────────────────────
    if "cooccur-hh" in steps:
        print("\n─── Шаг 5: Co-occurrence из HH-датасета ───")
        step_cooccur_hh()

    print()
    print("  [OK] Готово. Теперь можно запускать:")
    print("     python scorer.py 'финтех' 'банковские технологии'")
    print("     python demo.py")
    print("     python test_scenarios.py")


if __name__ == "__main__":
    main()
