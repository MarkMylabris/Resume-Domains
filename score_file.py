"""
score_file.py - Скоринг вакансий и резюме из текстовых файлов в реальном времени.

Принимает:
  --vacancy FILE        - текстовый файл с описанием вакансии (обязателен)
  --resume  FILE [...]  - один или несколько .txt файлов с текстом резюме
  --resume-dir DIR      - директория: каждый .txt-файл = одно резюме
  --top N               - вывести топ-N кандидатов (default: 10)
  --json                - вывод результата в JSON
  --out FILE            - сохранить вывод в файл

Если --resume / --resume-dir не указаны - используется готовый датасет
кандидатов (candidates_hh_full.json или candidates_normalized.json).

Примеры:
  python score_file.py --vacancy vacancy.txt
  python score_file.py --vacancy vacancy.txt --resume cv1.txt cv2.txt cv3.txt
  python score_file.py --vacancy vacancy.txt --resume-dir ./resumes/ --top 5
  python score_file.py --vacancy vacancy.txt --json --out result.json
"""

import sys, os, re, json, argparse
import numpy as np
from pathlib import Path

os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE = Path(__file__).parent
W    = 78

import scorer
from scorer import _load, _bar, explain_text
from live_demo import extract_domains_from_text, estimate_domain_weights
from parse_hh_resumes import extract_domains as extract_resume_domains, clean_resume


# ──────────────────────────────────────────────────────────────
#  ЗАГРУЗКА ФАЙЛОВ РЕЗЮМЕ
# ──────────────────────────────────────────────────────────────

def _load_resume_files(paths: list[Path]) -> dict[int, list[str]]:
    """Читает текстовые файлы резюме, извлекает домены из каждого.

    Возвращает {idx: [domains]}, где idx - порядковый номер (1-based).
    Имя файла запоминается в _resume_names для отображения.
    """
    global _resume_names
    _resume_names = {}
    candidates = {}

    print(f"\n  Извлечение доменов из {len(paths)} резюме...")
    for i, path in enumerate(paths, 1):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            print(f"  [WARN] {path.name} - пустой файл, пропущен")
            continue

        cleaned = clean_resume(text)
        if not cleaned:
            cleaned = text[:1000]   # fallback: берём сырой текст

        domains = extract_resume_domains(cleaned)
        candidates[i] = domains
        _resume_names[i] = path.stem   # имя файла без расширения

        pct = i * 100 // len(paths)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {i}/{len(paths)} ({pct}%)  {path.name[:30]:<30}", end="", flush=True)

    print()
    return candidates


_resume_names: dict[int, str] = {}


def _collect_resume_paths(args) -> list[Path]:
    """Собирает пути к .txt-файлам резюме из аргументов CLI."""
    paths = []
    if args.resume:
        for p in args.resume:
            fp = Path(p)
            if not fp.exists():
                print(f"  [WARN] Файл не найден: {fp}")
            else:
                paths.append(fp)
    if args.resume_dir:
        d = Path(args.resume_dir)
        if not d.is_dir():
            print(f"  [ERR] Директория не найдена: {d}")
            sys.exit(1)
        paths.extend(sorted(d.glob("*.txt")))
    return paths


# ──────────────────────────────────────────────────────────────
#  ЗАГРУЗКА ГОТОВОГО ДАТАСЕТА (fallback)
# ──────────────────────────────────────────────────────────────

_HH_FULL  = BASE / "datasets" / "hh_ru_resumes" / "candidates_hh_full.json"
_HH_SMALL = BASE / "datasets" / "hh_ru_resumes" / "candidates_hh.json"
_ORIG     = BASE / "candidates_normalized.json"


def _load_default_candidates(min_domains: int = 3) -> tuple[dict, str]:
    """Возвращает (candidates_dict, description).

    Фильтрует кандидатов с менее чем min_domains доменами - они неинформативны.
    min_domains=3 исключает профили без чёткой специализации.
    """
    for path, label in [
        (_HH_FULL,  "HH.ru полный (~23k)"),
        (_HH_SMALL, "HH.ru выборка (500)"),
        (_ORIG,     "тестовые 117 кандидатов"),
    ]:
        if path.exists():
            raw = json.loads(path.read_text())
            cands = {int(k): v for k, v in raw.items() if len(v) >= min_domains}
            return cands, f"{label} из {path.name} (>={min_domains} доменов)"
    print("  [ERR] Ни один файл кандидатов не найден. Укажите --resume или --resume-dir.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
#  ФОРМАТИРОВАНИЕ ВЫВОДА
# ──────────────────────────────────────────────────────────────

def _candidate_label(cid: int) -> str:
    """Отображаемое имя кандидата: имя файла или просто ID."""
    name = _resume_names.get(cid)
    return f'"{name}"' if name else f"#{cid}"


def _print_results(vacancy_text: str, query_domains: list[str],
                   results: list[dict], top_n: int, source_label: str) -> None:
    """Prints ranked candidates and compact explanations for file-based workflow.

    Input:
      - vacancy_text: raw vacancy text for header context.
      - query_domains: extracted vacancy domain list.
      - results: scorer output list.
      - top_n: number of top candidates to print.
      - source_label: human-readable label of candidate source.
    Output:
      - None. Writes formatted report to stdout.
    """
    sep = "─" * W
    print()
    print("═" * W)
    print(f"  Домены вакансии: {query_domains}")
    print(f"  Кандидатов в пуле: {source_label}")
    print("═" * W)
    print()
    print(f"  {'#':<4} {'Кандидат':<26} {'Покрытие':>8}  {'Оценка':<22}  Лучший домен")
    print("  " + sep)

    for rank_n, r in enumerate(results[:top_n], 1):
        cid  = r["id"]
        best = max(r["details"], key=lambda d: d["similarity"]) if r["details"] else {}
        hit  = "✓" if best.get("hit") else "✗"
        tx   = " +tax" if best.get("tax_bonus", 0) > 0 else ""
        match_str = (f"{hit} «{best.get('match','')[:20]}»  {best.get('similarity',0):.3f}{tx}"
                     if best else "-")
        label = _candidate_label(cid)
        print(f"  {rank_n:<4} {label:<26} {r['hit_rate']:>7.0%}  {_bar(r['score'])}  {match_str}")

    print("  " + sep)

    # Детальные профили топ-3
    print()
    for rank_n, r in enumerate(results[:min(3, top_n)], 1):
        cid   = r["id"]
        doms  = r["domains"]
        label = _candidate_label(cid)
        score = r["score"]
        hitrate = r["hit_rate"]

        print(f"  ┌─ #{rank_n} - {label}  (score={score:.3f}, покрытие={hitrate:.0%}) {'─'*max(1,W-40-len(label))}┐")
        main  = doms[:3]
        extra = doms[3:6]
        print(f"  │  Домены: " + " · ".join(f"«{d}»" for d in main))
        if extra:
            print(f"  │  Также:  " + " · ".join(f"«{d}»" for d in extra))
        print(f"  │  Вердикт: {explain_text(r)}")
        print(f"  │")
        hits   = [d for d in r["details"] if d["hit"]]
        misses = [d for d in r["details"] if not d["hit"]]
        for d in hits:
            tx     = f" (+tax {d['tax_bonus']:.2f})" if d.get("tax_bonus", 0) > 0 else ""
            w_mark = " (!)" if d.get("weight", 1.0) >= 1.8 else ""
            print(f"  │  ✓ «{d['query_domain'][:26]}»{w_mark} -> «{d['match'][:26]}»  {d['similarity']:.3f}{tx}")
        for d in misses[:2]:
            w_mark = " (!)" if d.get("weight", 1.0) >= 1.8 else ""
            print(f"  │  ✗ «{d['query_domain'][:26]}»{w_mark} -> «{d['match'][:26]}»  {d['similarity']:.3f}  (<0.72)")
        print(f"  └{'─'*(W-3)}┘")
        print()


def _to_json(query_domains: list[str], results: list[dict],
             top_n: int, source_label: str) -> str:
    """Serializes top ranking output to JSON string.

    Input:
      - query_domains: extracted vacancy domain list.
      - results: scorer output list.
      - top_n: number of candidates to include.
      - source_label: candidate source label.
    Output:
      - JSON string with ranking, domain matches, and explanations.
    """
    out = {
        "query_domains": query_domains,
        "candidates_source": source_label,
        "top": [],
    }
    for i, r in enumerate(results[:top_n]):
        out["top"].append({
            "rank":        i + 1,
            "id":          r["id"],
            "name":        _resume_names.get(r["id"], f"#{r['id']}"),
            "score":       round(r["score"], 4),
            "hit_rate":    round(r["hit_rate"], 4),
            "explanation": explain_text(r),
            "domains":     r["domains"],
            "matches":     [
                {
                    "query":      d["query_domain"],
                    "match":      d["match"],
                    "similarity": round(d["similarity"], 4),
                    "hit":        d["hit"],
                    "tax_bonus":  d.get("tax_bonus", 0),
                    "weight":     d.get("weight", 1.0),
                }
                for d in r["details"]
            ],
        })
    return json.dumps(out, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for scoring from vacancy/resume text files.

    Input:
      - Command-line arguments for vacancy path, resume files or directory, and output mode.
    Output:
      - None. Prints human-readable result or writes JSON file.
    """
    parser = argparse.ArgumentParser(
        description="Скоринг вакансии против резюме из текстовых файлов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python score_file.py --vacancy vacancy.txt
  python score_file.py --vacancy vacancy.txt --resume cv1.txt cv2.txt
  python score_file.py --vacancy vacancy.txt --resume-dir ./resumes/ --top 5
  python score_file.py --vacancy vacancy.txt --json --out result.json
""",
    )
    parser.add_argument("--vacancy",    required=True,      help="Файл с текстом вакансии (.txt)")
    parser.add_argument("--resume",     nargs="+",          help="Файлы резюме (.txt)")
    parser.add_argument("--resume-dir", metavar="DIR",      help="Директория с резюме (.txt)")
    parser.add_argument("--top",        type=int, default=10, help="Топ-N результатов (default: 10)")
    parser.add_argument("--json",       action="store_true", help="Вывод в JSON")
    parser.add_argument("--out",        metavar="FILE",      help="Сохранить результат в файл")
    args = parser.parse_args()

    # ── Читаем вакансию ────────────────────────────────────────
    vac_path = Path(args.vacancy)
    if not vac_path.exists():
        print(f"  [ERR] Файл вакансии не найден: {vac_path}")
        sys.exit(1)
    vacancy_text = vac_path.read_text(encoding="utf-8", errors="replace").strip()
    if not vacancy_text:
        print(f"  [ERR] Файл вакансии пуст: {vac_path}")
        sys.exit(1)

    # ── Инициализация модели ───────────────────────────────────
    print(f"\n  Инициализация модели...", end=" ", flush=True)
    _load()
    print("готово.")

    # ── Резюме: из файлов или из готового датасета ─────────────
    resume_paths = _collect_resume_paths(args)

    if resume_paths:
        candidates = _load_resume_files(resume_paths)
        source_label = f"{len(candidates)} резюме из файлов"
        scorer._candidates = candidates
    else:
        cands, desc = _load_default_candidates()
        scorer._candidates = cands
        source_label = desc
        print(f"  Кандидаты: {source_label}")

    # ── Извлечение доменов из вакансии ─────────────────────────
    print(f"\n  Извлечение доменов из вакансии ({vac_path.name})...")
    query_domains = extract_domains_from_text(vacancy_text, top_k=6)

    if not query_domains:
        print("  [ERR] Не удалось извлечь домены из вакансии. Проверьте текст файла.")
        sys.exit(1)

    # ── Определение важности доменов ───────────────────────────
    weights = estimate_domain_weights(vacancy_text, query_domains)
    _W_ICONS = {1.8: "[!]", 0.55: "[?]", 1.0: "[ ]"}
    for d in query_domains:
        w = weights.get(d, 1.0)
        print(f"    {_W_ICONS.get(w,'   ')} «{d}»  (w={w:.1f})")

    # ── Скоринг ────────────────────────────────────────────────
    print(f"\n  Скоринг {len(scorer._candidates)} кандидатов...", end=" ", flush=True)
    results = scorer.rank(query_domains, weights=weights)
    print("готово.")

    # ── Вывод ──────────────────────────────────────────────────
    if args.json:
        output = _to_json(query_domains, results, args.top, source_label)
    else:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_results(vacancy_text, query_domains, results, args.top, source_label)
        output = buf.getvalue()
        print(output)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"\n  ✓ Сохранено: {args.out}")


if __name__ == "__main__":
    main()
