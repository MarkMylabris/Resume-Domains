"""
demo_hh.py - Демо скоринга на реальных резюме из HH.ru датасета.

Использует те же вакансии из live_demo.py, но вместо 117 тестовых кандидатов
работает с ~23k реальными IT-резюме из HH.ru (candidates_hh_full.json).

Файл кандидатов выбирается автоматически:
  1. datasets/hh_ru_resumes/candidates_hh_full.json  (~23k, сбалансированный)
  2. datasets/hh_ru_resumes/candidates_hh.json        (500, fallback)

Запуск:
    python demo_hh.py               # все 8 вакансий
    python demo_hh.py --id 1 3      # вакансии #1 и #3
    python demo_hh.py --top 10      # топ-10 кандидатов
    python demo_hh.py --compare     # сравнение HH-пула vs оригинальных 117
"""

import sys, os, json, argparse
import numpy as np
from pathlib import Path
from collections import Counter

os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE    = Path(__file__).parent
# Предпочитаем полный сбалансированный датасет, fallback на меньший
HH_FILE = BASE / "datasets" / "hh_ru_resumes" / "candidates_hh_full.json"
if not HH_FILE.exists():
    HH_FILE = BASE / "datasets" / "hh_ru_resumes" / "candidates_hh.json"
W       = 78


# ──────────────────────────────────────────────────────────────
#  ЗАГРУЗКА
# ──────────────────────────────────────────────────────────────

import scorer
from scorer import _load, _bar, _norm, explain_text
import live_demo
from live_demo import VACANCIES, extract_domains_from_text, estimate_domain_weights


def _load_hh_candidates(min_domains: int = 3) -> dict:
    """Загружает HH-кандидатов, фильтрует слабые профили (<= min_domains-1 доменов).

    Кандидаты с 0-2 доменами неинформативны: они дают случайные низкие score
    почти для любого запроса и засоряют топ.
    min_domains=3 оставляет только профили с достаточной специализацией.
    """
    raw = json.loads(HH_FILE.read_text())
    return {int(k): v for k, v in raw.items() if len(v) >= min_domains}


def _init(use_hh: bool = True):
    """Инициализирует scorer, подменяет кандидатов если use_hh=True."""
    _load()
    if use_hh:
        scorer._candidates = _load_hh_candidates()
        print(f"  Кандидаты: HH.ru ({len(scorer._candidates)} резюме из {HH_FILE.name})")
    else:
        print(f"  Кандидаты: оригинальные ({len(scorer._candidates)} тестовых резюме)")


# ──────────────────────────────────────────────────────────────
#  ФОРМАТИРОВАНИЕ ПРОФИЛЯ HH-КАНДИДАТА
# ──────────────────────────────────────────────────────────────

def format_hh_profile(cid: int, domains: list[str]) -> str:
    """Читаемое описание профиля HH-кандидата по доменам."""
    role = live_demo.guess_role(domains)
    main  = domains[:3]
    extra = domains[3:6]
    lines = [f"  Профиль: {role}"]
    lines.append(f"  Домены:  " + " · ".join(f"«{d}»" for d in main))
    if extra:
        lines.append(f"  Также:   " + " · ".join(f"«{d}»" for d in extra))
    if len(domains) > 6:
        lines.append(f"           ... ещё {len(domains) - 6} доменов")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  СКОРИНГ ОДНОЙ ВАКАНСИИ НА HH-ПУЛЕ
# ──────────────────────────────────────────────────────────────

def run_vacancy(v: dict, top_n: int = 5) -> None:
    """Runs one vacancy scenario against HH candidate dataset and prints report.

    Input:
      - v: vacancy scenario dictionary with text, expected ids, and metadata.
      - top_n: number of top ranked candidates to display.
    Output:
      - None. Writes extraction, ranking, and explanation output to stdout.
    """
    vac_id   = v["id"]
    title    = v["title"]
    text     = v["text"]

    print()
    print("╔" + "═" * (W - 2) + "╗")
    pad = max(0, (W - 2 - len(title)) // 2)
    print("║" + " " * pad + title[:W-2] + " " * max(0, W - 2 - pad - len(title)) + "║")
    print("╚" + "═" * (W - 2) + "╝")
    print(f"\n  Компания: {v.get('company', '-')}")

    # Шаг 1: Извлечение доменов из текста вакансии
    print(f"\n  ● Шаг 1 - Извлечение доменов из текста вакансии (MMR)")
    extracted = extract_domains_from_text(text, top_k=6)

    # Шаг 2: Определение важности доменов
    print(f"\n  ● Шаг 2 - Важность доменов (эвристика + локальная модель)")
    weights = estimate_domain_weights(text, extracted)
    _W_ICONS = {1.8: "[!]", 0.55: "[?]", 1.0: "[ ]"}
    print(f"  Домены запроса:")
    for d in extracted:
        w = weights.get(d, 1.0)
        print(f"    {_W_ICONS.get(w,'   ')} «{d}»  (w={w:.1f})")

    # Шаг 3: Скоринг
    print(f"\n  ● Шаг 3 - Скоринг {len(scorer._candidates)} HH-кандидатов с учётом весов...")
    results = scorer.rank(extracted, weights=weights)

    # Топ кандидатов
    print(f"\n  ● Топ-{top_n} кандидатов из HH.ru\n")
    print(f"  {'#':<5} {'Кандидат':<11} {'Покрытие':>8}  {'Оценка':<22}  Лучший домен")
    print("  " + "─" * (W - 4))

    for rank_n, r in enumerate(results[:top_n], 1):
        cid  = r["id"]
        best = max(r["details"], key=lambda d: d["similarity"]) if r["details"] else {}
        mark = "✓" if best.get("hit") else "✗"
        tx   = " +tax" if best.get("tax_bonus", 0) > 0 else ""
        match_str = (f"{mark} «{best.get('match','')[:22]}»  {best.get('similarity',0):.3f}{tx}"
                     ) if best else "-"
        print(f"  {rank_n:<5} HH#{cid:<8} {r['hit_rate']:>7.0%}  {_bar(r['score'])}  {match_str}")

    print("  " + "─" * (W - 4))

    # Детальные профили топ-3
    print(f"\n  ● Профили топ-3 кандидатов:")
    for rank_n, r in enumerate(results[:3], 1):
        cid   = r["id"]
        doms  = r["domains"]
        score = r["score"]
        hitrate = r["hit_rate"]

        explanation = explain_text(r)
        print()
        print(f"  ┌─ #{rank_n} место - HH-кандидат #{cid}  "
              f"(score={score:.3f}, покрытие={hitrate:.0%}) {'─'*max(1,W-46-len(str(cid)))}┐")
        for line in format_hh_profile(cid, doms).splitlines():
            print(f"  │{line[1:]}")
        print(f"  │  Вердикт: {explanation}")
        print()
        hits   = [d for d in r["details"] if d["hit"]]
        misses = [d for d in r["details"] if not d["hit"]]
        if hits:
            for d in hits:
                tx = f" (+tax {d['tax_bonus']:.2f})" if d.get("tax_bonus", 0) > 0 else ""
                w_mark = " (!)" if d.get("weight", 1.0) >= 1.8 else ""
                print(f"  │  ✓ «{d['query_domain'][:26]}»{w_mark} -> «{d['match'][:26]}»  {d['similarity']:.3f}{tx}")
        if misses:
            for d in misses[:2]:
                w_mark = " (!)" if d.get("weight", 1.0) >= 1.8 else ""
                print(f"  │  ✗ «{d['query_domain'][:26]}»{w_mark} -> «{d['match'][:26]}»  {d['similarity']:.3f}  (<0.72)")
        print("  └" + "─" * (W - 3) + "┘")
    print()


# ──────────────────────────────────────────────────────────────
#  РЕЖИМ СРАВНЕНИЯ: HH-пул vs оригинальные 117
# ──────────────────────────────────────────────────────────────

def run_compare(vacancies: list[dict], top_n: int = 5) -> None:
    """Для каждой вакансии показывает топ из обоих пулов рядом."""
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + " " * 15 + "СРАВНЕНИЕ: HH-пул (500) vs Оригинал (117)" + " " * 20 + "║")
    print("╚" + "═" * (W - 2) + "╝")

    # Загружаем оба набора
    orig_cands = {int(k): v for k, v in
                  json.loads((BASE / "candidates_normalized.json").read_text()).items()}
    hh_cands   = _load_hh_candidates()

    for v in vacancies:
        title = v["title"]
        text  = v["text"]

        print(f"\n  ── {title[:60]} ──")
        extracted = extract_domains_from_text(text, top_k=6)
        print(f"  Домены: {extracted}")
        print()

        # Оригинальные кандидаты
        scorer._candidates = orig_cands
        orig_res = scorer.rank(extracted)[:top_n]

        # HH кандидаты
        scorer._candidates = hh_cands
        hh_res = scorer.rank(extracted)[:top_n]

        print(f"  {'Оригинал (117 кандидатов)':<38}  {'HH.ru (500 кандидатов)'}")
        print("  " + "─" * (W - 4))
        for i in range(top_n):
            o = orig_res[i] if i < len(orig_res) else None
            h = hh_res[i]   if i < len(hh_res)   else None
            o_str = f"#{o['id']:<4} {_bar(o['score'])} {o['score']:.3f}" if o else "-"
            h_str = f"HH#{h['id']:<4} {_bar(h['score'])} {h['score']:.3f}" if h else "-"
            print(f"  {i+1}. {o_str:<38}  {h_str}")
        print()


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for HH demo and optional baseline comparison mode.

    Input:
      - Command-line arguments (scenario ids, top, compare, no-expected).
    Output:
      - None. Prints scenario results and summary.
    """
    parser = argparse.ArgumentParser(
        description="Скоринг вакансий на HH.ru резюме",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python demo_hh.py                  # все 8 вакансий на HH-пуле
  python demo_hh.py --id 2 5         # вакансии #2 и #5
  python demo_hh.py --top 10         # топ-10 кандидатов
  python demo_hh.py --compare        # сравнение HH vs оригинал
  python demo_hh.py --compare --id 1 3
""",
    )
    parser.add_argument("--id",      nargs="+", type=int, help="ID вакансий (1–8)")
    parser.add_argument("--top",     type=int, default=5, help="Топ-N кандидатов")
    parser.add_argument("--compare", action="store_true",  help="Сравнить HH vs оригинал")
    args = parser.parse_args()

    if not HH_FILE.exists():
        print(f"\n  ✗ HH-датасет не найден: {HH_FILE}")
        print(f"  Запустите: python parse_hh_resumes.py --limit 500 --it-only")
        sys.exit(1)

    # Выбор вакансий
    vac_map  = {v["id"]: v for v in VACANCIES}
    vac_ids  = args.id or [v["id"] for v in VACANCIES]
    missing  = [i for i in vac_ids if i not in vac_map]
    if missing:
        print(f"  ✗ Вакансии {missing} не найдены. Доступны: {sorted(vac_map)}")
        sys.exit(1)
    vacancies = [vac_map[i] for i in vac_ids]

    print("\n  Инициализация модели...", end=" ", flush=True)
    _init(use_hh=not args.compare)   # compare сам переключает пулы
    print("готово.")

    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + " " * 10 + "DOMAIN SCORER - HH.RU КАНДИДАТЫ (реальные резюме)" + " " * 7 + "║")
    print("╚" + "═" * (W - 2) + "╝")

    if args.compare:
        run_compare(vacancies, top_n=args.top)
    else:
        for v in vacancies:
            run_vacancy(v, top_n=args.top)


if __name__ == "__main__":
    main()
