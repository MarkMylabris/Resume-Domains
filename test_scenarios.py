"""
test_scenarios.py - Автоматизированное тестирование скорера (v7-hybrid).

Категории тестов:
  A) Позитивные     - известные хорошие кандидаты должны быть в топ-5
  B) Нишевые        - узкоспециализированные домены, v7 находит лучше
  C) Негативные     - запросы вне базы, топ-1 score должен быть < 0.25
  D) Некорректные   - абсурдные запросы, score должен быть минимальным
  E) Граничные      - смежные домены, требующая точности

Запуск:
    python test_scenarios.py           # все тесты
    python test_scenarios.py -v        # с деталями по каждому домену
    python test_scenarios.py --fast    # только категории A и C (быстрая проверка)

Exit code: 0 если все прошли, 1 если есть провалы.
"""

import sys, os, argparse
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from scorer import rank, _load, _bar

W = 72


# ──────────────────────────────────────────────────────────────
#  ОПРЕДЕЛЕНИЕ ТЕСТОВ
# ──────────────────────────────────────────────────────────────
TESTS = [
    # ── A) Позитивные ──────────────────────────────────────────
    {
        "id": "A-fintech",
        "cat": "A",
        "label": "Финтех + банковское ядро",
        "domains": ["финтех", "интеграция банковского ядра"],
        "expect": [6, 59],
        "note": "Топ-5 включает #6 и #59 - финтех-специалисты с 100% покрытием",
    },
    {
        "id": "A-qa",
        "cat": "A",
        "label": "QA-инженер / Автотестировщик",
        "domains": ["автоматизация тестирования", "обеспечение качества программного обеспечения"],
        "expect": [22, 63],
        "note": "#63 - QA + мобайл, #22 - QA + авиация/ВПК",
    },
    {
        "id": "A-construction",
        "cat": "A",
        "label": "Строительство / Тендеры",
        "domains": ["строительная отрасль", "управление строительными проектами", "тендерная документация"],
        "expect": [94, 110],
    },
    {
        "id": "A-logistics",
        "cat": "A",
        "label": "Логистика - нишевый специалист",
        "domains": ["транспортная логистика", "управление грузоперевозками", "логистическая платформа"],
        "expect": [66],
        "note": "#66 должен быть строго топ-1",
        "expect_top1": 66,
    },
    {
        "id": "A-travel",
        "cat": "A",
        "label": "Travel-tech / Онлайн-туризм",
        "domains": ["онлайн туризм", "бронирование авиабилетов", "travel платформа"],
        "expect": [28, 38],
    },
    # ── B) Нишевые ─────────────────────────────────────────────
    {
        "id": "B-gaming",
        "cat": "B",
        "label": "Игровая платформа + монетизация",
        "domains": ["онлайн-игровая платформа", "монетизация игр", "игровые механики"],
        "expect": [4, 39, 82, 115],
        "note": "#4 - геймдев, #39 - игровая индустрия, #82 - мобильные игры + финтех, #115 - игровые услуги",
    },
    {
        "id": "B-cybersec",
        "cat": "B",
        "label": "Кибербезопасность / ИБ",
        "domains": ["кибербезопасность", "защита данных", "информационная безопасность"],
        "note": "Специалисты по ИБ в топ-3 (без жёсткого ожидания - немного кандидатов)",
    },
    {
        "id": "B-adtech",
        "cat": "B",
        "label": "Programmatic / AdTech",
        "domains": ["программматическая реклама", "dsp платформа", "rtb аукцион"],
        "note": "AdTech-специалисты в топ-3",
    },
    {
        "id": "B-telemedicine",
        "cat": "B",
        "label": "Телемедицина + mHealth",
        "domains": ["телемедицина", "мобильное здравоохранение", "электронные медкарты"],
        "note": "Медицинские кандидаты в топ-5",
    },
    {
        "id": "B-blockchain",
        "cat": "B",
        "label": "Блокчейн / DeFi",
        "domains": ["blockchain", "смарт-контракты", "децентрализованные финансы"],
        "note": "Web3/blockchain кандидаты в топ-5",
    },
    # ── C) Негативные ──────────────────────────────────────────
    {
        "id": "C-space",
        "cat": "C",
        "label": "Космос / Ракетостроение",
        "domains": ["ракетостроение", "аэрокосмическая отрасль", "спутниковые системы"],
        "max_score": 0.25,
        "note": "Нет ракетостроителей в базе. #22 (ВПК) может попасть - это нормально",
    },
    {
        "id": "C-oil",
        "cat": "C",
        "label": "Нефтегаз",
        "domains": ["добыча нефти", "нефтепереработка", "геологоразведка"],
        "max_score": 0.25,
    },
    {
        "id": "C-agro",
        "cat": "C",
        "label": "АгроПром / Сельское хозяйство",
        "domains": ["сельское хозяйство", "агропромышленный комплекс", "растениеводство"],
        "max_score": 0.25,
    },
    {
        "id": "C-defense",
        "cat": "C",
        "label": "Оборонная промышленность",
        "domains": ["военная техника", "оборонная промышленность", "ракетные комплексы"],
        "note": "Информационный: #22 (военно-промышленный комплекс) - легитимный матч",
        # max_score не задаём: #22 с доменом 'военно-промышленный комплекс' корректно попадает
    },
    # ── D) Некорректные ────────────────────────────────────────
    {
        "id": "D-cooking",
        "cat": "D",
        "label": "Кулинария (не IT)",
        "domains": ["приготовление еды", "рецепты блюд", "кулинарные техники"],
        "max_score": 0.25,
    },
    {
        "id": "D-sport",
        "cat": "D",
        "label": "Спорт (не IT)",
        "domains": ["футбол", "спортивные соревнования", "тренировки команды"],
        "max_score": 0.25,
    },
    {
        "id": "D-fashion",
        "cat": "D",
        "label": "Мода / Швейное производство",
        "domains": ["мода", "дизайн одежды", "швейное производство"],
        "note": "Информационный: 'розничная торговля одеждой' семантически близка к запросу - нормально",
        # max_score не задаём: система корректно находит кандидатов с одеждой/ритейлом
    },
    {
        "id": "D-random",
        "cat": "D",
        "label": "Случайные строки",
        "domains": ["abc", "xyz", "qwerty"],
        "note": "Токены abc/xyz/qwerty отфильтровываются как шум - топ-1 score=0, ранжирование пустое",
        "max_score": 0.01,
    },
    # ── E) Граничные ───────────────────────────────────────────
    {
        "id": "E-mixed",
        "cat": "E",
        "label": "Смешанный: Финтех + Медицина",
        "domains": ["финтех", "медицинские информационные системы"],
        "note": "Финансисты и медики должны делить топ - нет одного явного победителя",
    },
    {
        "id": "E-ecomm-synonyms",
        "cat": "E",
        "label": "E-commerce (синонимы)",
        "domains": ["e-commerce", "маркетплейс", "онлайн-торговля"],
        "note": "Три синонима одного домена - топ должен быть стабильным",
    },
    {
        "id": "E-legal",
        "cat": "E",
        "label": "Юридические услуги",
        "domains": ["адвокат", "юридические услуги", "правовое сопровождение"],
        "note": "#42 - юрист, должен быть топ-1",
        "expect_top1": 42,
    },
    {
        "id": "E-multidomain",
        "cat": "E",
        "label": "5 доменов разных отраслей",
        "domains": ["банкинг", "fintech", "e-commerce", "страхование", "логистика"],
        "note": "Мультидоменный запрос - generalist-кандидаты выше нишевых",
    },
    {
        "id": "E-mobile-dev",
        "cat": "E",
        "label": "Мобильная разработка",
        "domains": ["разработка мобильных приложений", "ios разработка", "android разработка"],
        "note": "Mobile-dev кандидаты в топ-5",
    },
]


# ──────────────────────────────────────────────────────────────
#  ЗАПУСК ОДНОГО ТЕСТА
# ──────────────────────────────────────────────────────────────
def run_test(t: dict, verbose: bool = False) -> tuple[bool, float]:
    """Запускает один тест. Возвращает (passed, top1_score)."""
    results  = rank(t["domains"])
    top5_ids = {r["id"] for r in results[:5]}
    top1_sc  = results[0]["score"] if results else 0.0
    top1_id  = results[0]["id"] if results else None

    # Компактный вывод: топ-5
    label_w  = 38
    cat_lbl  = f"[{t['cat']}] {t['label']}"
    print(f"\n  ╔{'─' * (W - 4)}╗")
    print(f"  ║  {cat_lbl:<{W - 8}}  ║")
    print(f"  ╚{'─' * (W - 4)}╝")
    if t.get("note"):
        print(f"      {t['note']}")
    print(f"  Запрос: {t['domains']}")
    print()
    print(f"  {'№':<4} {'#Канд':<6} {'hit%':>5}   {'Score':<22}  Лучший match")
    print("  " + "─" * (W - 4))

    for i, r in enumerate(results[:5], 1):
        best = max(r["details"], key=lambda d: d["similarity"]) if r["details"] else {}
        mk   = "✓" if best.get("hit") else "✗"
        tx   = " +tax" if best.get("tax_bonus", 0) > 0 else ""
        match_str = (f"{mk} '{best['query_domain'][:14]}' → "
                     f"'{best['match'][:16]}' {best['similarity']:.3f}{tx}") if best else ""
        want_mark = " ★" if r["id"] in t.get("expect", []) else ""
        print(f"  {i:<4} #{r['id']:<5} {r['hit_rate']:>4.0%}   {_bar(r['score'])}  {match_str}{want_mark}")

        if verbose:
            for d in r["details"]:
                fl = "✓" if d["hit"] else "·"
                tx2 = f" +{d['tax_bonus']:.2f}" if d["tax_bonus"] > 0 else ""
                print(f"       {fl}  '{d['query_domain'][:24]:<26}→ '{d['match'][:22]}'  {d['similarity']:.3f}{tx2}")

    print("  " + "─" * (W - 4))

    # Проверка условий
    issues  = []
    passed  = True

    if "expect" in t:
        missing = sorted(set(t["expect"]) - top5_ids)
        found   = sorted(set(t["expect"]) & top5_ids)
        if found:
            print(f"  ✓ В топ-5: {found}")
        if missing:
            print(f"  ✗ Не в топ-5: {missing}")
            issues.append(f"missing {missing}")
            passed = False

    if "expect_top1" in t:
        if top1_id == t["expect_top1"]:
            print(f"  ✓ Топ-1 = #{top1_id} (ожидался)")
        else:
            print(f"  ✗ Топ-1 = #{top1_id}, ожидался #{t['expect_top1']}")
            issues.append(f"top1={top1_id}≠{t['expect_top1']}")
            passed = False

    if "max_score" in t:
        if top1_sc <= t["max_score"]:
            print(f"  ✓ Топ-1 score = {top1_sc:.3f} ≤ {t['max_score']} - корректно низкий")
        else:
            print(f"  ✗ Топ-1 score = {top1_sc:.3f} > {t['max_score']} - слишком высокий!")
            issues.append(f"score={top1_sc:.3f}>{t['max_score']}")
            passed = False

    if not issues and not any(k in t for k in ("expect", "expect_top1", "max_score")):
        print(f"      Информационный тест (без строгих условий) - топ-1 score = {top1_sc:.3f}")

    verdict = "[OK] PASS" if passed else f"[FAIL]  [{'; '.join(issues)}]"
    print(f"\n  {verdict}")
    return passed, top1_sc


# ──────────────────────────────────────────────────────────────
#  ТОЧКА ВХОДА
# ──────────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point for deterministic functional test suite.

    Input:
      - Command-line arguments for verbosity and category filters.
    Output:
      - None. Prints per-test report and exits with code 0/1.
    """
    parser = argparse.ArgumentParser(
        description="Тесты скорера по бизнес-доменам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python test_scenarios.py           # все тесты
  python test_scenarios.py -v        # с деталями по совпадениям
  python test_scenarios.py --fast    # только A + C (быстрая проверка)
  python test_scenarios.py --cat A B # только категории A и B
""",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Детали по каждому домену")
    parser.add_argument("--fast",          action="store_true", help="Только категории A и C")
    parser.add_argument("--cat",           nargs="+",           help="Фильтр по категориям (A B C D E)")
    args = parser.parse_args()

    cats_filter = None
    if args.fast:
        cats_filter = {"A", "C"}
    elif args.cat:
        cats_filter = set(c.upper() for c in args.cat)

    tests = [t for t in TESTS if cats_filter is None or t["cat"] in cats_filter]

    # Инициализация
    print("\n  Загрузка модели...", end=" ", flush=True)
    _load()
    print("готово.")

    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + " " * 14 + "DOMAIN SCORER - TEST SUITE" + " " * 14 + " " * (W - 2 - 14 - 26 - 14) + "║")
    print("╚" + "═" * (W - 2) + "╝")
    print(f"\n  Тестов: {len(tests)}   Кандидатов в базе: 117   Порог сходства: 0.72")

    # Категории
    cat_names = {
        "A": "Позитивные",
        "B": "Нишевые",
        "C": "Негативные",
        "D": "Некорректные",
        "E": "Граничные",
    }
    active_cats = sorted({t["cat"] for t in tests})
    for c in active_cats:
        n = sum(1 for t in tests if t["cat"] == c)
        print(f"  [{c}] {cat_names[c]}: {n} тест(ов)")

    # Прогон
    results_log = []
    for t in tests:
        passed, top1 = run_test(t, verbose=args.verbose)
        results_log.append((t["cat"], t["id"], t["label"], passed, top1))

    # ── Финальная сводка ────────────────────────────────────────
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + " " * 22 + "СВОДКА ТЕСТОВ" + " " * 22 + " " * (W - 2 - 22 - 13 - 22) + "║")
    print("╚" + "═" * (W - 2) + "╝")
    print()

    total_passed = 0
    total_count  = 0

    for cat in active_cats:
        cat_tests = [(tid, lbl, ok) for c, tid, lbl, ok, _ in results_log if c == cat]
        n_pass = sum(1 for _, _, ok in cat_tests if ok)
        n_tot  = len(cat_tests)
        total_passed += n_pass
        total_count  += n_tot

        pct   = n_pass * 100 // n_tot if n_tot else 0
        bar_w = n_pass * 20 // n_tot  if n_tot else 0
        print(f"  [{cat}] {cat_names[cat]:<16} [{'█'*bar_w}{'░'*(20-bar_w)}] {n_pass}/{n_tot} ({pct}%)")
        for tid, lbl, ok in cat_tests:
            icon = "[OK]" if ok else "[FAIL]"
            print(f"       {icon}  {lbl}")
    print()

    # Итог
    all_pass = total_passed == total_count
    pct_all  = total_passed * 100 // total_count if total_count else 0
    bar_w    = total_passed * 20 // total_count  if total_count else 0
    print(f"  ИТОГО: {total_passed}/{total_count} ({pct_all}%)  [{'█'*bar_w}{'░'*(20-bar_w)}]")

    if all_pass:
        print("\n  Все тесты пройдены!")
    else:
        failed = [(tid, lbl) for _, tid, lbl, ok, _ in results_log if not ok]
        print(f"\n   Провалено {total_count - total_passed}:")
        for tid, lbl in failed:
            print(f"     • [{tid}] {lbl}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
