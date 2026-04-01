"""
demo.py - Интерактивная демонстрация скорера по бизнес-доменам.

Запуск:
    python demo.py                  # показать все предустановленные сценарии
    python demo.py --scenario qa    # запустить один сценарий по имени
    python demo.py --interactive    # ввести свой запрос вручную
    python demo.py --list           # список всех сценариев
"""

import sys, os, argparse
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from scorer import rank, explain, _bar, _load

W = 72


# ──────────────────────────────────────────────────────────────
#  ПРЕДУСТАНОВЛЕННЫЕ СЦЕНАРИИ
# ──────────────────────────────────────────────────────────────
SCENARIOS = {
    # Позитивные - кандидаты должны быть в топ
    "fintech": {
        "label":   "Финтех + банковское ядро",
        "domains": ["финтех", "интеграция банковского ядра"],
        "note":    "Классический финтех-специалист: #35 (топ-1), #6, #59 в топ-5",
        "expect":  [35, 6, 59],
    },
    "qa": {
        "label":   "QA-инженер / Автотестировщик",
        "domains": ["автоматизация тестирования", "обеспечение качества программного обеспечения"],
        "note":    "Специалисты по тестированию: #63 (топ-1), #22 в топ-5",
        "expect":  [63, 22],
    },
    "logistics": {
        "label":   "Логистика / Грузоперевозки",
        "domains": ["транспортная логистика", "управление грузоперевозками", "логистическая платформа"],
        "note":    "Нишевый специалист: #66 должен быть топ-1",
        "expect":  [66],
    },
    "construction": {
        "label":   "Строительство / Тендеры",
        "domains": ["строительная отрасль", "управление строительными проектами", "тендерная документация"],
        "note":    "Строительные компании: #110, #94 в топ-5",
        "expect":  [94, 110],
    },
    "travel": {
        "label":   "Travel-tech / Онлайн-туризм",
        "domains": ["онлайн туризм", "бронирование авиабилетов", "travel платформа"],
        "note":    "Туристические платформы: #28, #38 в топ-5",
        "expect":  [28, 38],
    },
    "gaming": {
        "label":   "Игровая платформа + монетизация",
        "domains": ["онлайн-игровая платформа", "монетизация игр", "игровые механики"],
        "note":    "Игровая индустрия: #4 (топ-1), #39, #82, #115 в топ-5",
        "expect":  [4, 39, 82, 115],
    },
    "kafka": {
        "label":   "Apache Kafka + Микросервисы",
        "domains": ["apache kafka", "микросервисная архитектура", "event-driven системы"],
        "note":    "Бэкенд-специалисты с брокерами сообщений",
    },
    "cybersec": {
        "label":   "Кибербезопасность / ИБ",
        "domains": ["кибербезопасность", "защита данных", "информационная безопасность"],
        "note":    "Специалисты по информационной безопасности: #112 в топ-1",
    },
    "ecommerce": {
        "label":   "E-commerce / Маркетплейс",
        "domains": ["e-commerce", "маркетплейс", "онлайн-торговля"],
        "note":    "Синонимы одного домена - должен собирать похожих кандидатов",
    },
    # Негативные - нет кандидатов, топ-1 должен быть низким
    "space": {
        "label":   "Космос / Ракетостроение (нет в базе)",
        "domains": ["ракетостроение", "аэрокосмическая отрасль", "спутниковые системы"],
        "note":    "Негативный тест: ни один кандидат не является ракетчиком. Топ-1 < 0.25",
        "max_score": 0.25,
    },
    "oil": {
        "label":   "Нефтегаз (нет в базе)",
        "domains": ["добыча нефти", "нефтепереработка", "геологоразведка"],
        "note":    "Негативный тест: топ-1 должен быть < 0.25",
        "max_score": 0.25,
    },
    # Некорректные - абсурдные запросы
    "cooking": {
        "label":   "Кулинария (не IT)",
        "domains": ["приготовление еды", "рецепты блюд", "кулинарные техники"],
        "note":    "Некорректный запрос: топ-1 должен быть < 0.25",
        "max_score": 0.25,
    },
}


# ──────────────────────────────────────────────────────────────
#  ФОРМАТИРОВАНИЕ ВЫВОДА
# ──────────────────────────────────────────────────────────────
def section(title: str, char: str = "═") -> None:
    """Prints a decorative section header for terminal demo output.

    Input:
      - title: section title text.
      - char: border character.
    Output:
      - None. Writes header lines to stdout.
    """
    pad = max(0, (W - 2 - len(title)) // 2)
    print()
    print("╔" + char * (W - 2) + "╗")
    print("║" + " " * pad + title + " " * (W - 2 - pad - len(title)) + "║")
    print("╚" + char * (W - 2) + "╝")


def print_scenario(key: str, sc: dict, top: int = 5) -> bool:
    """Выводит результаты одного сценария. Возвращает True если прошёл проверку."""
    section(f"[{key.upper()}] {sc['label']}", "─")
    print(f"  {sc['note']}")
    print(f"  Домены запроса: {sc['domains']}")
    print()

    results = rank(sc["domains"])
    top5_ids = {r["id"] for r in results[:5]}

    print(f"  {'№':<5} {'#Канд':<7} {'hit%':>5}   {'Score':<24}  Лучший match")
    print("  " + "─" * (W - 4))

    for i, r in enumerate(results[:top], 1):
        best = max(r["details"], key=lambda d: d["similarity"]) if r["details"] else {}
        mark = "✓" if best.get("hit") else "✗"
        tx   = " +tax" if best.get("tax_bonus", 0) > 0 else ""
        match_info = (f"{mark} '{best['query_domain'][:13]}' → "
                      f"'{best['match'][:16]}' {best['similarity']:.3f}{tx}") if best else ""
        star = " ★" if r["id"] in sc.get("expect", set()) else ""
        print(f"  {i:<5} #{r['id']:<6} {r['hit_rate']:>4.0%}   {_bar(r['score'])}  {match_info}{star}")

    print("  " + "─" * (W - 4))

    # Вердикт
    passed = True
    if "expect" in sc:
        found   = sorted(set(sc["expect"]) & top5_ids)
        missing = sorted(set(sc["expect"]) - top5_ids)
        if found:
            print(f"  ✓ Ожидаемые кандидаты в топ-5: {found}")
        if missing:
            print(f"  ✗ Не найдены в топ-5:           {missing}")
            passed = False

    if "max_score" in sc:
        top1 = results[0]["score"]
        if top1 <= sc["max_score"]:
            print(f"  ✓ Топ-1 score = {top1:.3f} ≤ {sc['max_score']} - корректно низкий")
        else:
            print(f"  ✗ Топ-1 score = {top1:.3f} > {sc['max_score']} - слишком высокий!")
            passed = False

    status = "[OK]" if passed else "[FAIL] ПРОВЕРЬТЕ"
    print(f"\n  {status}")
    return passed


# ──────────────────────────────────────────────────────────────
#  ИНТЕРАКТИВНЫЙ РЕЖИМ
# ──────────────────────────────────────────────────────────────
def interactive_mode() -> None:
    """Runs interactive query mode for manual domain input testing.

    Input:
      - Reads user input from stdin line-by-line.
    Output:
      - None. Prints ranked candidates for each query.
    """
    section("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("  Введите домены вакансии через запятую или по одному (пустая строка = выход).")
    print('  Пример: финтех, банкинг, платёжные системы')
    print()

    while True:
        try:
            raw = input("  > Домены запроса: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Выход.")
            break

        if not raw:
            break

        # Поддерживаем разделители: запятая или перевод строки
        domains = [d.strip() for d in raw.replace(",", "\n").splitlines() if d.strip()]
        if not domains:
            continue

        try:
            n_raw = input("  > Показать топ (Enter = 5): ").strip()
            top_n = int(n_raw) if n_raw.isdigit() else 5
        except (EOFError, KeyboardInterrupt):
            top_n = 5

        results = rank(domains)
        print()
        print(f"  {'№':<5} {'#Канд':<7} {'hit%':>5}   {'Score':<24}  Домены кандидата")
        print("  " + "─" * (W - 4))
        for i, r in enumerate(results[:top_n], 1):
            best = max(r["details"], key=lambda d: d["similarity"]) if r["details"] else {}
            mark = "✓" if best.get("hit") else "✗"
            match_info = (f"{mark} '{best['query_domain'][:13]}' → "
                          f"'{best['match'][:16]}' {best['similarity']:.3f}") if best else ""
            print(f"  {i:<5} #{r['id']:<6} {r['hit_rate']:>4.0%}   {_bar(r['score'])}  {match_info}")
            # Показываем домены кандидата
            dom_str = ", ".join(r["domains"][:4])
            if len(r["domains"]) > 4:
                dom_str += f" (+{len(r['domains'])-4})"
            print(f"  {'':5} {'':7} {'':5}    {'':24}  [{dom_str}]")
        print("  " + "─" * (W - 4))
        print()


# ──────────────────────────────────────────────────────────────
#  ТОЧКА ВХОДА
# ──────────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point for predefined and interactive demo modes.

    Input:
      - Command-line arguments for scenario id and interactive mode.
    Output:
      - None. Prints demo ranking output.
    """
    parser = argparse.ArgumentParser(
        description="Demo - демонстрация скорера по бизнес-доменам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python demo.py                      # все сценарии
  python demo.py --scenario fintech   # один сценарий
  python demo.py --scenario qa gaming # несколько сценариев
  python demo.py --interactive        # интерактивный ввод
  python demo.py --list               # список сценариев
""",
    )
    parser.add_argument("--scenario",    nargs="*",   help="Ключи сценариев (из --list)")
    parser.add_argument("--interactive", action="store_true", help="Интерактивный режим")
    parser.add_argument("--list",        action="store_true", help="Показать список сценариев")
    parser.add_argument("--top",         type=int, default=5,  help="Топ-N кандидатов")
    args = parser.parse_args()

    if args.list:
        print("\n  Доступные сценарии:\n")
        for key, sc in SCENARIOS.items():
            kind = "🔴 негатив" if "max_score" in sc else "🟢 позитив"
            print(f"  {key:<15} {kind}  -  {sc['label']}")
            print(f"  {'':15}              {sc['domains']}")
            print()
        return

    # Инициализация модели (один раз)
    print("\n  Загрузка модели...", end=" ", flush=True)
    _load()
    print("готово.")

    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + " " * 15 + "DOMAIN SCORER - DEMO" + " " * 15 + " " * (W - 2 - 15 - 20 - 15) + "║")
    print("╚" + "═" * (W - 2) + "╝")

    if args.interactive:
        interactive_mode()
        return

    # Выбираем сценарии
    if args.scenario is not None:
        keys = args.scenario if args.scenario else list(SCENARIOS.keys())
        missing = [k for k in keys if k not in SCENARIOS]
        if missing:
            print(f"  Неизвестные сценарии: {missing}. Используйте --list.")
            sys.exit(1)
    else:
        keys = list(SCENARIOS.keys())

    passed_count = 0
    for key in keys:
        ok = print_scenario(key, SCENARIOS[key], top=args.top)
        if ok:
            passed_count += 1

    # Итог
    total = len(keys)
    checkable = sum(1 for k in keys if "expect" in SCENARIOS[k] or "max_score" in SCENARIOS[k])
    if checkable:
        section("ИТОГ")
        bar_w = passed_count * 20 // total if total else 0
        print(f"\n  Проверяемых сценариев: {checkable}/{total}")
        print(f"  Прошло:  {passed_count}/{total}  {'█' * bar_w}{'░' * (20 - bar_w)}")
        if passed_count == total:
            print("\n  [OK] Все сценарии в норме!")
        else:
            print(f"\n   {total - passed_count} сценарий(ев) требует внимания.")


if __name__ == "__main__":
    main()
