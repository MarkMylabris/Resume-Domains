"""
live_demo.py - Демо на живых текстах вакансий.

Показывает полный конвейер:
  1. Реальный текст вакансии (неструктурированный)
  2. Автоматическое извлечение бизнес-доменов (семантический поиск по словарю)
  3. Скоринг всех 117 кандидатов
  4. Топ-5 с «резюме» - читаемым описанием профиля кандидата

Извлечение доменов из текста:
  - Вакансия кодируется целиком → ищем ближайшие домены в словаре
  - Применяем MMR (Maximal Marginal Relevance) для разнообразия
  - Результат: 4–6 ключевых доменов, передаются в scorer

Запуск:
    python live_demo.py              # все вакансии
    python live_demo.py --id 2       # одна вакансия по номеру
    python live_demo.py --text "..."  # произвольный текст
    python live_demo.py --interactive # интерактивный ввод
"""

import sys, os, re, json, argparse
import numpy as np
from pathlib import Path

os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import scorer                                     # модуль целиком - обращаемся через него
from scorer import rank, _load, _bar, EXPANSIONS  # только статические объекты

BASE = Path(__file__).parent
W    = 76


# ──────────────────────────────────────────────────────────────
#  РЕАЛЬНЫЕ ВАКАНСИИ - живые тексты, как приходят от HR
# ──────────────────────────────────────────────────────────────
VACANCIES = [
    {
        "id": 1,
        "title": "Senior Backend Engineer - Финтех / Банковская интеграция",
        "company": "FinCore Solutions",
        "text": """
Мы разрабатываем платформу для интеграции с банковскими ядрами (АБС) российских и
зарубежных банков. В команде используем Python и Go, архитектура - микросервисная на Kafka.

Ищем опытного бэкенд-инженера, который:
  - имеет опыт в финтехе или банковском секторе от 4 лет
  - работал с интеграцией банковского ядра (Flextera, ЦФТ, Diasoft)
  - знаком с платёжными системами, SWIFT, СБП, НСПК
  - понимает процессы кредитования, расчётных операций банков
  - умеет проектировать высоконагруженные системы

Будет плюсом: опыт с KYC/AML процессами, открытый банкинг (Open Banking API).
Удалённо, полная занятость. Зарплата от 350 000 руб.
        """.strip(),
        "expected_cands": [59, 6, 35],
    },
    {
        "id": 2,
        "title": "QA Automation Engineer - мобильные приложения",
        "company": "AppTest Lab",
        "text": """
Нам нужен специалист по автоматизированному тестированию. Мы тестируем мобильные
приложения (iOS/Android) и веб-платформы для клиентов из разных отраслей.

Обязательные требования:
  • Опыт автоматизации тестирования от 3 лет (Selenium, Appium, pytest)
  • Глубокое понимание QA-процессов и обеспечения качества ПО
  • Навыки API-тестирования (Postman, REST Assured)
  • Опыт управления дефектами в Jira
  • SQL для верификации данных

Желательно: опыт в финтех или e-commerce проектах, знание CI/CD пайплайнов.
Офис - Москва, Сити. Гибкий график.
        """.strip(),
        "expected_cands": [63, 22],
    },
    {
        "id": 3,
        "title": "Product Manager - Игровая платформа / Монетизация",
        "company": "PlayVenture Studio",
        "text": """
Растущая игровая студия ищет Product Manager для развития онлайн-платформы.
Наши продукты: казуальные мобильные игры с механиками социального взаимодействия,
in-app монетизация, battle pass, loot boxes.

Чем предстоит заниматься:
  - Развитие игровой платформы и экосистемы (онлайн-игровая платформа)
  - Проектирование и A/B-тестирование игровых механик
  - Рост монетизации через freemium и микротранзакции
  - Работа с геймдизайнерами над progression loop и game economy
  - Анализ retention, LTV, ARPU для мобильных игр

Ищем человека с пониманием игровой индустрии, желателен опыт в геймдеве или
смежных областях (беттинг, казино, развлекательные платформы).
        """.strip(),
        "expected_cands": [4, 39, 82, 115],
    },
    {
        "id": 4,
        "title": "Head of Logistics Tech - Цифровизация грузоперевозок",
        "company": "FreightDigital",
        "text": """
Федеральный логистический оператор переходит к цифровой платформе управления
перевозками. Ищем технического руководителя направления.

Задачи:
  * Разработка стратегии цифровизации транспортной логистики
  * Управление командой разработки логистической платформы
  * Автоматизация процессов: заявки, маршрутизация, трекинг грузов
  * Интеграция с ТК, экспедиторами, складскими системами WMS
  * KPI: скорость доставки, стоимость перевозки, SLA

Опыт: управление грузоперевозками или транспортной логистикой от 5 лет,
знание TMS-систем, понимание B2B-рынка логистики.
        """.strip(),
        "expected_cands": [66],
    },
    {
        "id": 5,
        "title": "Chief Information Security Officer (CISO)",
        "company": "SecureBank Group",
        "text": """
Банковская группа ищет руководителя службы информационной безопасности.

Ответственность:
  - Стратегия и архитектура кибербезопасности группы
  - Управление SOC, incident response, penetration testing
  - Соответствие требованиям ЦБ РФ, ФСТЭК, 152-ФЗ
  - Защита данных клиентов, DLP, SIEM, IAM
  - Работа с регуляторами по вопросам информационной безопасности

Требования: опыт в ИБ от 7 лет, знание стандартов ISO 27001 / PCI DSS,
понимание финансовых рисков и специфики банковского сектора.
        """.strip(),
        "expected_cands": [112, 48],
    },
    {
        "id": 6,
        "title": "E-commerce Director - Маркетплейс / Онлайн-ритейл",
        "company": "MegaShop Marketplace",
        "text": """
Крупный маркетплейс ищет директора по e-commerce для управления ростом платформы.

Направления работы:
  • Стратегия роста GMV и числа продавцов на маркетплейсе
  • P&L онлайн-торговли: интернет-торговля, категорийный менеджмент
  • Развитие логистики последней мили и фулфилмента
  • CRM, retention, работа с лояльностью покупателей
  • Запуск новых вертикалей: fashion, электроника, продукты питания

Ищем человека с глубоким пониманием e-commerce и опытом управления
крупными онлайн-платформами. Будет плюсом: опыт в ритейле или FMCG.
        """.strip(),
    },
    {
        "id": 7,
        "title": "Fullstack Developer - EdTech / Образовательная платформа",
        "company": "LearnFast",
        "text": """
Стартап в сфере онлайн-образования ищет fullstack-разработчика для создания
адаптивной платформы персонализированного обучения.

Стек: React, Node.js, PostgreSQL, Redis, Docker. Контент-система с LMS,
видеотрансляции, интерактивные курсы, тестирование знаний.

Задачи:
  - Разработка личного кабинета студента и преподавателя
  - Интеграция с платёжными системами (подписки, единоразовые оплаты)
  - Видеостриминг и работа с медиаконтентом
  - Геймификация: бейджи, рейтинги, достижения
  - Нагрузочное тестирование платформы

Команда - 12 человек, продукт выходит на рынок СНГ. Удалённо.
        """.strip(),
    },
    {
        "id": 8,
        "title": "Главный юрисконсульт - банковское и налоговое право",
        "company": "LexFinance",
        "text": """
Юридическая фирма, специализирующаяся на финансовом праве, ищет главного юриста.

Ключевые компетенции:
  • Правовое сопровождение банковских кредитных операций
  • Налоговое планирование, возврат НДС, налоговые споры
  • Кредитное сопровождение корпоративных клиентов
  • Юридическая экспертиза финансовых инструментов
  • Представление интересов клиентов в судах и налоговых органах

Обязательно: адвокатский статус или опыт в налоговом / банковском праве от 6 лет.
        """.strip(),
        "expected_cands": [42],
    },
]


# ──────────────────────────────────────────────────────────────
#  ИЗВЛЕЧЕНИЕ ДОМЕНОВ ИЗ ТЕКСТА (семантический поиск + MMR)
# ──────────────────────────────────────────────────────────────

# Домены, которые семантически «тянутся» к большинству текстов, но не несут
# бизнес-специфики нужной для разграничения вакансий по областям.
_VACANCY_BLOCKLIST = {
    # Слишком широкие управленческие домены - ложно матчатся к любой IT-вакансии с PM-функцией
    "управление проектами в машиностроении",   # GameDev PM, Backend PM -> ложное
    "управление строительными объектами",       # аналогично
    "управление производством в строительстве",
    # Государственные/госсервисы - не бизнес-домены для коммерческих вакансий
    "цифровой ассистент госуслуг",
    "государственные информационные услуги",
    "личный кабинет государственных услуг",
    "автоматизация бизнес-процессов государственных услуг",
    # Слишком технически общие
    "mes система",
    "веб-браузер",
    "конструктор административных панелей",
    "панель администратора",
    "управление корзиной",
    "интерактивные доски в онлайн-образовании",
    "цифровая трансформация бизнес-процессов",
    "проведение ido",
}


def extract_domains_from_text(text: str, top_k: int = 6,
                               mmr_lambda: float = 0.65) -> list[str]:
    """Находит top_k наиболее релевантных доменов из словаря для произвольного текста.

    Использует MMR (Maximal Marginal Relevance) для разнообразия:
      score(d) = lambda * sim(d, query) - (1-lambda) * max_sim(d, already_selected)
    Это позволяет избежать выбора 5 вариаций одного понятия.
    Обращается к scorer._model/_emb_v7/_domain_list через модуль (они заполняются после _load()).
    """
    import contextlib, io
    _buf = io.StringIO()
    with contextlib.redirect_stderr(_buf):
        text_emb = scorer._model.encode([text], normalize_embeddings=True)[0]

    emb   = scorer._emb_v7
    dlist = scorer._domain_list

    # Индексы заблокированных доменов
    blocked = {i for i, d in enumerate(dlist) if d in _VACANCY_BLOCKLIST}

    # Сходство текста со всеми доменами
    sims = emb @ text_emb                          # dot-product = cosine (L2-norm=1)
    candidates_idx = [i for i in np.argsort(sims)[::-1][:120] if i not in blocked]

    # MMR-выбор - итеративно добавляем самый релевантный и непохожий на уже выбранных
    selected_idx, selected_embs = [], []
    for _ in range(top_k):
        best_score, best_i = -1.0, -1
        for ci in candidates_idx:
            if ci in selected_idx:
                continue
            relevance  = float(sims[ci])
            redundancy = max((float(emb[ci] @ e) for e in selected_embs), default=0.0)
            score = mmr_lambda * relevance - (1 - mmr_lambda) * redundancy
            if score > best_score:
                best_score, best_i = score, ci
        if best_i == -1:
            break
        selected_idx.append(best_i)
        selected_embs.append(emb[best_i])

    return [dlist[i] for i in selected_idx]


# ──────────────────────────────────────────────────────────────
#  ОПРЕДЕЛЕНИЕ ВАЖНОСТИ ДОМЕНОВ (ЭВРИСТИКА + ЛОКАЛЬНАЯ МОДЕЛЬ)
# ──────────────────────────────────────────────────────────────

# Заголовки секций «обязательных» требований
_REQUIRED_HEADERS = frozenset([
    "обязательн", "обязателен", "требования", "что нужно", "что мы ищем",
    "ждём от вас", "ждем от вас", "ищем кандидата", "необходим",
    "required", "mandatory", "must have", "hard skills", "key requirements",
    "о кандидате", "от кандидата", "hard skill",
])
# Заголовки / слова секций «желательных» требований
_OPTIONAL_HEADERS = frozenset([
    "желательн", "будет плюсом", "будет преимуществом", "будет большим плюсом",
    "nice to have", "preferred", "приветствуется", "приветствуем если",
    "дополнительн", "не обязательн", "плюсом будет", "soft skills",
])
# Заголовки нейтральных секций (условия, описание компании - НЕ требования)
_NEUTRAL_HEADERS = frozenset([
    "мы предлагаем", "что мы предлагаем", "условия работы", "условия",
    "о компании", "о нас", "мы", "компания предлагает", "what we offer",
    "benefits", "обязанности", "задачи", "что предстоит делать", "функции",
    "ваши задачи", "what you'll do", "responsibilities",
])

# Пороговая разница сходства для уверенного отнесения к группе
_WEIGHT_GAP = 0.05
_W_REQUIRED = 1.8   # «обязательно» - существенно влияет на score
_W_OPTIONAL = 0.55  # «желательно» - меньший вес, но не ноль
_W_NEUTRAL  = 1.0


def _parse_vacancy_sections(text: str) -> tuple[list[str], list[str]]:
    """Разбивает текст вакансии на секции «обязательного» и «желательного».

    Стратегия:
      1. Детектируем заголовки секций (строки <80 симв. с ключевыми словами).
      2. Все пункты/строки под required-заголовком → required_lines.
         Под optional-заголовком → optional_lines.
         Под neutral-заголовком → игнорируем.
      3. Строки до первого заголовка считаем телом вакансии (implicit required).
      4. Fallback: если заголовки не найдены, ищем inline-сигналы.
    """
    req_lines: list[str] = []
    opt_lines: list[str] = []

    current = "body"   # body / required / optional / neutral

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower().rstrip(":").rstrip(".")

        # Определяем, является ли строка заголовком секции
        is_short = len(line) < 80
        is_req   = is_short and any(h in low for h in _REQUIRED_HEADERS)
        is_opt   = is_short and any(h in low for h in _OPTIONAL_HEADERS)
        is_neu   = is_short and any(h in low for h in _NEUTRAL_HEADERS)

        # Смена режима по заголовку (optional проверяем первым - он перекрывает required)
        # Если заголовок и контент на одной строке ("Будет плюсом: KYC/AML"),
        # извлекаем контент после двоеточия и добавляем в соответствующий список.
        if is_opt:
            colon_pos = line.find(":")
            if 0 < colon_pos < len(line) - 3:
                tail = line[colon_pos + 1:].strip()
                if len(tail) > 8:
                    opt_lines.append(tail)
            current = "optional"
            continue
        if is_req and not is_neu:
            colon_pos = line.find(":")
            if 0 < colon_pos < len(line) - 3:
                tail = line[colon_pos + 1:].strip()
                if len(tail) > 8:
                    req_lines.append(tail)
            current = "required"
            continue
        if is_neu and not is_req:
            current = "neutral"
            continue

        # Накапливаем строки по текущей секции
        if len(line) < 10:   # слишком короткий маркер/символ - пропускаем
            continue
        if current in ("body", "required"):
            req_lines.append(line)
        elif current == "optional":
            opt_lines.append(line)
        # neutral игнорируем

    # Fallback: если заголовки не дали секций, ищем inline-сигналы
    if not opt_lines:
        for line in text.splitlines():
            low = line.strip().lower()
            if any(h in low for h in _OPTIONAL_HEADERS) and len(line.strip()) > 15:
                opt_lines.append(line.strip())

    return req_lines, opt_lines


def estimate_domain_weights(
    vacancy_text: str,
    domains: list[str],
) -> dict[str, float]:
    """Оценивает важность каждого домена: секционный парсинг + локальная модель.

    Алгоритм:
      1. Разбиваем текст по секциям («Требования:», «Будет плюсом:» и т.д.).
         Строки до первого заголовка = тело вакансии = implicit required.
      2. Если optional-секции нет → нет смысла взвешивать, все = 1.0.
      3. Кодируем строки секций и домены локальной sentence-transformer моделью.
      4. Вес домена:
           1.8 если он ближе к required-секции (или нет optional-секции)
           0.55 если он ближе к optional-секции (выше порога abs_sim)
           1.0  иначе (неопределённо)

    Возвращает dict {domain: weight}.
    """
    import contextlib, io

    req_lines, opt_lines = _parse_vacancy_sections(vacancy_text)

    # Если нет optional-секции - нельзя ничего дифференцировать
    if not opt_lines:
        return {d: _W_NEUTRAL for d in domains}

    # Кодируем домены с расширениями для лучшего семантического соотнесения
    def _domain_text(d: str) -> str:
        exp = EXPANSIONS.get(d, "")
        return (exp[:120] if exp and exp != d else d)

    texts_to_encode = req_lines + opt_lines + [_domain_text(d) for d in domains]
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        all_embs = scorer._model.encode(
            texts_to_encode, normalize_embeddings=True, show_progress_bar=False
        )

    req_embs = all_embs[:len(req_lines)]
    opt_embs = all_embs[len(req_lines): len(req_lines) + len(opt_lines)]
    dom_embs = all_embs[len(req_lines) + len(opt_lines):]

    # Если req-секция пустая - нельзя разграничить required vs optional
    # Используем повышенный порог: помечаем optional только явно специфичные
    _opt_abs_threshold = 0.38 if len(req_lines) > 0 else 0.52

    weights = {}
    for i, d in enumerate(domains):
        de = dom_embs[i]
        req_sim = float(np.max(req_embs @ de)) if len(req_embs) else 0.0
        opt_sim = float(np.max(opt_embs @ de)) if len(opt_embs) else 0.0

        if req_sim > opt_sim + _WEIGHT_GAP:
            w = _W_REQUIRED
        elif opt_sim > req_sim + _WEIGHT_GAP and opt_sim >= _opt_abs_threshold:
            w = _W_OPTIONAL
        else:
            w = _W_NEUTRAL
        weights[d] = w

    return weights


# ──────────────────────────────────────────────────────────────
#  ФОРМАТИРОВАНИЕ «РЕЗЮМЕ» КАНДИДАТА
# ──────────────────────────────────────────────────────────────
_DOMAIN_CATEGORIES = {
    # Ключевые слова для угадывания «роли» кандидата
    "qa": ["тестирование", "qa", "качества", "дефект", "автоматизация тест", "баг"],
    "fintech": ["финтех", "банкинг", "банков", "платёж", "кредит", "страхов"],
    "gamedev": ["игр", "геймд", "беттинг", "казино", "стриминг"],
    "logistics": ["логистик", "перевозк", "транспорт", "склад"],
    "ecommerce": ["e-commerce", "маркетплейс", "ритейл", "торговл"],
    "security": ["безопасност", "кибер", "информационная безопас"],
    "medtech": ["медицин", "здравоохран", "телемедицина"],
    "legal": ["правов", "юридич", "налог", "кредит сопровожд"],
}

_ROLE_LABELS = {
    "qa": "QA / Тестирование",
    "fintech": "Финтех / Банкинг",
    "gamedev": "Геймдев / iGaming",
    "logistics": "Логистика",
    "ecommerce": "E-commerce / Ритейл",
    "security": "Информационная безопасность",
    "medtech": "Медтех / Healthtech",
    "legal": "Юридические услуги",
}


def guess_role(domains: list[str]) -> str:
    """Infers a candidate role label from extracted domain list.

    Input:
      - domains: list of candidate business domains.
    Output:
      - Short role label string.
    """
    text = " ".join(domains).lower()
    scores = {}
    for role, keywords in _DOMAIN_CATEGORIES.items():
        scores[role] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)
    return _ROLE_LABELS.get(best, "IT / Технологии") if scores[best] > 0 else "IT / Технологии"


def format_resume(cid: int, domains: list[str]) -> str:
    """Форматирует домены кандидата как краткое резюме-профиль."""
    role = guess_role(domains)
    lines = [f"  Профиль: {role}"]

    # Группируем домены - первые 3 «главные», остальные - «дополнительные»
    main   = domains[:3]
    extra  = domains[3:7]

    lines.append(f"  Опыт:    " + " · ".join(f"«{d}»" for d in main))
    if extra:
        lines.append(f"  Также:   " + " · ".join(f"«{d}»" for d in extra))
    if len(domains) > 7:
        lines.append(f"           ... ещё {len(domains) - 7} областей")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  ВЫВОД ОДНОЙ ВАКАНСИИ
# ──────────────────────────────────────────────────────────────
def section(title: str, char: str = "═") -> None:
    """Prints a visual section header in demo output.

    Input:
      - title: section caption text.
      - char: frame character for top and bottom line.
    Output:
      - None. Writes formatted text to stdout.
    """
    pad = max(0, (W - 2 - len(title)) // 2)
    print()
    print("╔" + char * (W - 2) + "╗")
    print("║" + " " * pad + title + " " * (W - 2 - pad - len(title)) + "║")
    print("╚" + char * (W - 2) + "╝")


def run_vacancy(v: dict, top_n: int = 5, show_extraction: bool = True) -> None:
    """Обрабатывает одну вакансию и выводит результат."""
    vac_id   = v["id"]
    title    = v["title"]
    text     = v["text"]
    expected = v.get("expected_cands", [])

    section(f"[#{vac_id}] {title}", "═")
    print(f"\n  Компания: {v.get('company', '-')}")

    # ── Текст вакансии ──
    print()
    print("  ┌─ ТЕКСТ ВАКАНСИИ " + "─" * (W - 20) + "┐")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Переносим длинные строки
        while len(line) > W - 6:
            cut = line[:W - 6].rfind(" ")
            if cut < 0:
                cut = W - 6
            print(f"  │  {line[:cut]}")
            line = line[cut:].strip()
        print(f"  │  {line}")
    print("  └" + "─" * (W - 3) + "┘")

    # ── Извлечение доменов ──
    print()
    print("  ● Шаг 1 - Автоматическое извлечение бизнес-доменов из текста (MMR)")
    extracted = extract_domains_from_text(text, top_k=6)

    # ── Важность доменов ──
    print()
    print("  ● Шаг 2 - Определение важности доменов (эвристика + локальная модель)")
    weights = estimate_domain_weights(text, extracted)

    print()
    print("  Извлечённые домены и их важность:")
    _W_ICONS = {_W_REQUIRED: "[обязательно]", _W_OPTIONAL: "[желательно]", _W_NEUTRAL: "[нейтрально]"}
    for i, d in enumerate(extracted, 1):
        w    = weights.get(d, 1.0)
        icon = _W_ICONS.get(w, f"[w={w:.1f}]")
        exp  = EXPANSIONS.get(d, "")
        hint = f"  [{exp[:40]}]" if exp and exp != d else ""
        print(f"    {i}. «{d}»  {icon}{hint}")

    # ── Скоринг ──
    n_cands = len(scorer._candidates)
    print()
    print(f"  ● Шаг 3 - Скоринг {n_cands} кандидатов по доменам с учётом весов")
    results = rank(extracted, weights=weights)

    # ── Топ кандидаты ──
    print()
    print("  ● Шаг 4 - Топ кандидатов")
    print()
    print(f"  {'Место':<5} {'Кандидат':<11} {'Покрытие':>8}  {'Оценка':<22}  Лучший домен")
    print("  " + "─" * (W - 4))

    for rank_n, r in enumerate(results[:top_n], 1):
        cid  = r["id"]
        best = max(r["details"], key=lambda d: d["similarity"]) if r["details"] else {}
        mark = "✓" if best.get("hit") else "✗"
        tx   = " +tax" if best.get("tax_bonus", 0) > 0 else ""
        want_mark = " ●" if cid in expected else ""
        match_str = (f"{mark} «{best.get('match','')[:22]}»  {best.get('similarity',0):.3f}{tx}"
                     ) if best else ""
        print(f"  {rank_n:<5} #{cid:<10} {r['hit_rate']:>7.0%}  {_bar(r['score'])}  {match_str}{want_mark}")

    print("  " + "─" * (W - 4))

    # ── Детальный профиль топ-3 ──
    print()
    print("  ● Профили топ-3 кандидатов:")

    for rank_n, r in enumerate(results[:3], 1):
        cid    = r["id"]
        doms   = r["domains"]
        is_exp = cid in expected

        star = " ★ ОЖИДАЕМЫЙ" if is_exp else ""
        explanation = scorer.explain_text(r)
        print()
        print(f"  ┌─ #{rank_n} место - Кандидат #{cid}{star}  "
              f"(score={r['score']:.3f}, покрытие={r['hit_rate']:.0%}) {'─'*max(1,W-50-len(str(cid))-len(star))}┐")
        for line in format_resume(cid, doms).splitlines():
            print(f"  │{line[1:]}")
        print(f"  │  Вердикт: {explanation}")
        print()
        # Детали совпадений
        hits   = [d for d in r["details"] if d["hit"]]
        misses = [d for d in r["details"] if not d["hit"]]
        if hits:
            for d in hits:
                tx = f" (+tax {d['tax_bonus']:.2f})" if d.get("tax_bonus", 0) > 0 else ""
                w_icon = "(!)" if d.get("weight", 1.0) >= _W_REQUIRED else ""
                print(f"  │  ✓ «{d['query_domain'][:26]}»{w_icon}  ->  «{d['match'][:26]}»  {d['similarity']:.3f}{tx}")
        if misses:
            for d in misses[:2]:
                w_icon = " (!)" if d.get("weight", 1.0) >= _W_REQUIRED else ""
                print(f"  │  ✗ «{d['query_domain'][:26]}»{w_icon}  ->  «{d['match'][:26]}»  {d['similarity']:.3f}  (ниже 0.72)")
        print("  └" + "─" * (W - 3) + "┘")

    # ── Вердикт ──
    if expected:
        top5_ids  = {r["id"] for r in results[:5]}
        found     = sorted(set(expected) & top5_ids)
        missing   = sorted(set(expected) - top5_ids)
        print()
        if found:
            print(f"  [OK] Ожидаемые кандидаты в топ-5: {found}")
        if missing:
            print(f"   Не вошли в топ-5: {missing}")
    print()


# ──────────────────────────────────────────────────────────────
#  ИНТЕРАКТИВНЫЙ РЕЖИМ
# ──────────────────────────────────────────────────────────────
def interactive_mode(top_n: int = 5) -> None:
    """Runs interactive vacancy analysis loop in terminal mode.

    Input:
      - top_n: number of top candidates to display per request.
    Output:
      - None. Interacts with user via stdin/stdout.
    """
    section("ИНТЕРАКТИВНЫЙ РЕЖИМ - введите текст вакансии")
    print("  Вставьте текст вакансии (несколько строк, закончите пустой строкой):")
    print()

    while True:
        lines = []
        try:
            while True:
                line = input("  > ")
                if line == "":
                    if lines:
                        break
                else:
                    lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\n  Выход.")
            break

        if not lines:
            break

        text = "\n".join(lines)
        v = {"id": "?", "title": "Пользовательский запрос", "text": text, "company": "-"}
        run_vacancy(v, top_n=top_n)

        again = input("  Ввести ещё одну вакансию? [y/N]: ").strip().lower()
        if again != "y":
            break


# ──────────────────────────────────────────────────────────────
#  ТОЧКА ВХОДА
# ──────────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point for live demo scenarios and interactive mode.

    Input:
      - Command-line arguments (id, text, interactive, top, no-expected).
    Output:
      - None. Prints demo results and summary.
    """
    parser = argparse.ArgumentParser(
        description="Live Demo - скоринг на реальных текстах вакансий",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python live_demo.py               # все 8 вакансий
  python live_demo.py --id 1 2      # вакансии #1 и #2
  python live_demo.py --id 3 --top 7
  python live_demo.py --text "Ищем Python-разработчика для финтех-стартапа..."
  python live_demo.py --interactive
""",
    )
    parser.add_argument("--id",          nargs="+", type=int, help="Номера вакансий (1–8)")
    parser.add_argument("--text",        type=str,            help="Произвольный текст вакансии")
    parser.add_argument("--top",         type=int, default=5, help="Топ-N кандидатов (по умолчанию 5)")
    parser.add_argument("--interactive", action="store_true", help="Интерактивный ввод")
    args = parser.parse_args()

    # Инициализация
    print("\n  Инициализация модели...", end=" ", flush=True)
    _load()
    print("готово.")

    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + " " * 10 + "DOMAIN SCORER - LIVE DEMO (реальные тексты)" + " " * 11 + "║")
    print("╚" + "═" * (W - 2) + "╝")

    if args.interactive:
        interactive_mode(top_n=args.top)
        return

    if args.text:
        v = {"id": "?", "title": "Произвольная вакансия", "text": args.text, "company": "-"}
        run_vacancy(v, top_n=args.top)
        return

    # Выбираем вакансии
    if args.id:
        vac_map = {v["id"]: v for v in VACANCIES}
        missing = [i for i in args.id if i not in vac_map]
        if missing:
            print(f"  ✗ Вакансии с ID {missing} не существуют. Доступны: 1–{len(VACANCIES)}")
            sys.exit(1)
        vacancies = [vac_map[i] for i in args.id]
    else:
        vacancies = VACANCIES

    for v in vacancies:
        run_vacancy(v, top_n=args.top)

    # Итоговая сводка если несколько вакансий
    if len(vacancies) > 1:
        section("ИТОГОВАЯ СВОДКА")
        print()
        checked = [v for v in vacancies if v.get("expected_cands")]
        if checked:
            from scorer import rank as _rank
            ok_count = 0
            for v in checked:
                r5 = {r["id"] for r in _rank(
                    extract_domains_from_text(v["text"])
                )[:5]}
                found = set(v["expected_cands"]) & r5
                miss  = set(v["expected_cands"]) - r5
                icon  = "[OK]" if not miss else ""
                if not miss:
                    ok_count += 1
                status = f"найдены {sorted(found)}" + (f", пропущены {sorted(miss)}" if miss else "")
                print(f"  {icon}  #{v['id']} {v['title'][:40]:<40} {status}")
            print()
            print(f"  Вакансий с ожиданиями: {len(checked)}, совпали полностью: {ok_count}/{len(checked)}")


if __name__ == "__main__":
    main()
