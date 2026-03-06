import os
import re
import sqlite3
import struct
import json
from collections import defaultdict
from pathlib import Path
from colorama import init, Fore, Style

init(autoreset=True)

# ================= НАСТРОЙКИ =================
INDEX_DIR = Path("./data_contentless")
CSV_DIR = Path("./data_csv")
MAX_RESULTS = 30
BATCH_SIZE = 5
# =============================================


def normalize_query(q: str) -> str:
    """Нормализует поисковый запрос"""
    q = q.lower().strip()
    q = re.sub(r"[^а-яa-z0-9@._\- ]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q


def fts_build_exact_query(q: str) -> str:
    """Строит точный поисковый запрос для FTS5 - все слова должны быть"""
    parts = re.split(r"[^а-яa-z0-9@._\-]+", q.lower())
    parts = [p for p in parts if p]
    
    if not parts:
        return ""
    
    if len(parts) == 1:
        # Для одного слова - точное совпадение в начале или целое слово
        return f'"{parts[0]}"*'
    
    # Для нескольких слов - все должны быть в одной записи (AND)
    query = " AND ".join([f'"{p}"*' for p in parts])
    return query


def get_line_from_csv(csv_path: Path, offsets_path: Path, line_num: int, encoding: str = "utf-8") -> str:
    """Читает строку по номеру из CSV, используя .offsets.bin"""
    if not offsets_path.exists():
        return ""

    try:
        offset_idx = line_num - 1
        if offset_idx < 0:
            return ""

        with offsets_path.open("rb") as off_f:
            off_f.seek(offset_idx * 8)
            offset_bytes = off_f.read(8)
            if not offset_bytes or len(offset_bytes) < 8:
                return ""
            offset = struct.unpack("Q", offset_bytes)[0]

        with csv_path.open("rb") as f:
            f.seek(offset)
            line = f.readline().decode(encoding, errors="replace").rstrip("\n\r")

        return line
    except Exception:
        return ""


def load_metadata(metadata_path: Path) -> dict:
    """Загружает метаданные структуры CSV"""
    if not metadata_path.exists():
        return {}
    
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_csv_line(line: str, delimiter: str) -> list:
    """Парсит строку CSV"""
    return [p.strip() for p in line.split(delimiter)]


def calculate_relevance(raw_line: str, query_parts: list, col_indices: dict, delimiter: str) -> int:
    """
    Вычисляет релевантность найденной записи.
    Выше баллы для совпадений в более важных полях.
    """
    parts = parse_csv_line(raw_line, delimiter)
    score = 0
    
    # Приоритет полей (важные поля дают больше очков)
    field_priority = {
        'phone': 100,
        'uid': 80,
        'first_name': 90,
        'last_name': 90,
        'full_name': 85,
        'email': 70,
        'gender': 20,
        'location': 30,
        'hometown': 25,
        'birth_date': 10,
        'created_at': 5,
    }
    
    for query_part in query_parts:
        query_lower = query_part.lower()
        
        # Проверяем каждое важное поле
        for field_name, priority in field_priority.items():
            if field_name not in col_indices:
                continue
                
            col_idx = col_indices[field_name]
            if col_idx >= len(parts):
                continue
            
            value = parts[col_idx].lower()
            
            # Полное совпадение - максимум очков
            if value == query_lower:
                score += priority * 3
            # Совпадение в начале
            elif value.startswith(query_lower):
                score += priority * 2
            # Совпадение где-то внутри
            elif query_lower in value:
                score += priority
    
    return score


def format_structured_record(raw_line: str, metadata: dict, db_name: str) -> str:
    """Форматирует запись с полями на основе метаданных - красивый вывод"""
    lines = [f"{Fore.MAGENTA}{'═' * 70}"]
    lines.append(f"{Fore.GREEN}✓ Найден результат{Style.RESET_ALL}")
    lines.append(f"{Fore.CYAN}📦 Источник:{Style.RESET_ALL} {db_name}")
    lines.append("")

    delimiter = metadata.get('delimiter', ',')
    headers = metadata.get('headers', [])
    col_indices = metadata.get('column_indices', {})

    # Парсим строку
    parts = parse_csv_line(raw_line, delimiter)

    # Порядок и форматирование полей
    field_display = [
        ('phone', '📱', 'Phone'),
        ('uid', '🆔', 'UID'),
        ('first_name', '👤', 'First Name'),
        ('last_name', '👤', 'Last Name'),
        ('full_name', '👤', 'Name'),
        ('email', '📧', 'Email'),
        ('gender', '⚧', 'Gender'),
        ('location', '📍', 'Location'),
        ('hometown', '🏠', 'Hometown'),
        ('birth_date', '🎂', 'Birth Date'),
        ('created_at', '📅', 'Created'),
    ]

    # Выводим структурированные данные
    for field_name, emoji, label in field_display:
        if field_name in col_indices:
            col_idx = col_indices[field_name]
            if col_idx < len(parts):
                value = parts[col_idx].strip()
                # Пропускаем пустые значения и даты по умолчанию
                if value and value != '0001-01-01' and value != '1970-01-01':
                    lines.append(f"{Fore.YELLOW}{emoji} {label}:{Style.RESET_ALL} {value}")

    lines.append("")
    lines.append(f"{Fore.BLUE}📄 Raw:{Style.RESET_ALL}")
    raw_preview = raw_line[:100] + "..." if len(raw_line) > 100 else raw_line
    lines.append(f"{Fore.WHITE}{raw_preview}{Style.RESET_ALL}")
    
    lines.append(f"{Fore.MAGENTA}{'─' * 70}{Style.RESET_ALL}\n")

    return "\n".join(lines)


def search_db_optimized(index_db: Path, query: str, csv_path: Path, db_name: str, encoding: str, indexed_fields: list) -> list:
    """Поиск в оптимизированной FTS5 БД с сортировкой по релевантности"""
    offsets_path = index_db.with_suffix('.offsets.bin')
    metadata_path = index_db.with_suffix('.meta.json')

    if not index_db.exists():
        return []

    metadata = load_metadata(metadata_path)
    col_indices = metadata.get('column_indices', {})

    conn = sqlite3.connect(index_db)
    cur = conn.cursor()

    q = normalize_query(query)
    fts_q = fts_build_exact_query(q)

    if not fts_q:
        conn.close()
        return []

    # Разбиваем запрос на части для подсчета релевантности
    query_parts = re.split(r"[^а-яa-z0-9@._\-]+", q.lower())
    query_parts = [p for p in query_parts if p]

    try:
        # FTS5 поиск - ищет с точностью (все слова должны быть)
        cur.execute(
            "SELECT rowid FROM records WHERE records MATCH ? LIMIT ?",
            (fts_q, BATCH_SIZE * 50)  # Берем больше для сортировки
        )
        rowids = [r[0] for r in cur.fetchall()]

        results = []
        for rid in rowids:
            raw_line = get_line_from_csv(csv_path, offsets_path, rid, encoding)
            if raw_line:
                # Вычисляем релевантность
                relevance = calculate_relevance(raw_line, query_parts, col_indices, metadata.get('delimiter', ','))
                results.append((rid, raw_line, db_name, metadata_path, relevance))

        # Сортируем по релевантности (по убыванию)
        results.sort(key=lambda x: x[4], reverse=True)
        
        # Удаляем скор из результата
        results = [(r[0], r[1], r[2], r[3]) for r in results]

        return results

    except Exception as e:
        print(f"{Fore.RED}⚠ Ошибка в {db_name}: {e}{Style.RESET_ALL}")
        return []
    finally:
        conn.close()


def search_all(query: str) -> list:
    """Поиск по всем доступным базам"""
    all_results = []

    db_list = [f.stem for f in INDEX_DIR.glob("*.db") if f.suffix == ".db"]

    for db_name in db_list:
        db_path = INDEX_DIR / f"{db_name}.db"
        csv_path = CSV_DIR / f"{db_name}.csv"
        
        if not db_path.exists():
            continue
        
        if not csv_path.exists():
            csv_candidates = list(CSV_DIR.glob(f"{db_name}*"))
            if not csv_candidates:
                continue
            csv_path = csv_candidates[0]
        
        try:
            import chardet
            with csv_path.open("rb") as f:
                raw = f.read(100_000)
            result = chardet.detect(raw)
            encoding = result.get("encoding") or "utf-8"
        except Exception:
            encoding = "utf-8"
        
        # Загружаем метаданные для получения списка индексированных полей
        metadata_path = INDEX_DIR / f"{db_name}.meta.json"
        metadata = load_metadata(metadata_path)
        indexed_fields = metadata.get('indexed_fields', ['phone', 'uid', 'first_name', 'last_name', 'email', 'gender'])
        
        res = search_db_optimized(db_path, query, csv_path, db_name, encoding, indexed_fields)
        all_results.extend(res)

    # Глобальная сортировка по релевантности (если много БД)
    return all_results


def print_header():
    """Выводит красивый заголовок"""
    print(f"\n{Fore.MAGENTA}{'═' * 70}")
    print(f"{Fore.MAGENTA}      🔎 Точный поиск FTS5")
    print(f"{Fore.MAGENTA}{'═' * 70}\n")


def main():
    print_header()
    print(f"{Fore.CYAN}✓ Режим: поиск по всем доступным базам\n{Style.RESET_ALL}")

    while True:
        query = input(
            f"{Fore.GREEN}Введите поисковый запрос (или 'exit' для выхода):{Style.RESET_ALL} "
        ).strip()

        if query.lower() == "exit":
            print(f"\n{Fore.MAGENTA}👋 До свидания!{Style.RESET_ALL}\n")
            break

        if not query:
            print(f"{Fore.YELLOW}⚠ Пожалуйста, введите запрос{Style.RESET_ALL}")
            continue

        print(f"\n{Fore.CYAN}🔍 Поиск '{query}'...{Style.RESET_ALL}")

        results = search_all(query)

        if not results:
            print(f"\n{Fore.RED}❌ Ничего не найдено{Style.RESET_ALL}\n")
            continue

        print(f"\n{Fore.CYAN}📊 Результаты:{Style.RESET_ALL}")
        print(f"   • Найдено: {Fore.YELLOW}{len(results)}{Style.RESET_ALL}\n")

        # Выводим результаты
        for i, (rid, raw_line, db_name, metadata_path) in enumerate(results[:MAX_RESULTS], 1):
            metadata = load_metadata(metadata_path)
            output = format_structured_record(raw_line, metadata, db_name)
            print(output)

        remaining = len(results) - MAX_RESULTS
        if remaining > 0:
            print(f"{Fore.YELLOW}... и ещё {remaining} результатов{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
