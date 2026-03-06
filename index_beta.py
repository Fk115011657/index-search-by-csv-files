import sys
import os
import csv
import sqlite3
import time
import chardet
import logging
import struct
import json
from pathlib import Path
from io import StringIO

csv.field_size_limit(sys.maxsize)

# ================= НАСТРОЙКИ =================
CSV_DIR = Path("./data_csv")
INDEX_DIR = Path("./data_contentless")
LOG_DIR = Path("./logs")

BATCH_SIZE = 10000
COMMIT_EVERY = 500_000
PROGRESS_EVERY = 1_000_000

# Поля для индексирования (только эти будут в FTS5)
INDEXED_FIELDS = ['phone', 'uid', 'first_name', 'last_name', 'email', 'gender']

# =============================================

INDEX_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"contentless_index_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def detect_encoding(file_path: Path) -> str:
    """Определяет кодировку файла"""
    try:
        with file_path.open("rb") as f:
            raw = f.read(100_000)
        result = chardet.detect(raw)
        encoding = result.get("encoding") or "utf-8"
        try:
            "test".encode(encoding)
            return encoding
        except LookupError:
            return "utf-8"
    except Exception:
        return "utf-8"


def detect_delimiter(file_path: Path, encoding: str, sample_lines=50) -> str:
    """Определяет разделитель (,;|\t:)"""
    candidates = [',', ';', '\t', '|', ':']
    counts = {c: 0 for c in candidates}
    
    try:
        with file_path.open("r", encoding=encoding, errors="replace") as f:
            for _ in range(sample_lines):
                line = f.readline().rstrip("\n\r")
                if not line:
                    break
                if line.startswith("\ufeff"):
                    line = line.lstrip("\ufeff")
                for c in candidates:
                    counts[c] += line.count(c)
    except Exception as e:
        logger.debug(f"Ошибка при определении разделителя: {e}")
        return ','
    
    if not counts or all(v == 0 for v in counts.values()):
        return ','
    
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ','


def parse_csv_line(line: str, delimiter: str) -> list:
    """Правильно парсит CSV строку с обработкой кавычек"""
    try:
        reader = csv.reader(StringIO(line), delimiter=delimiter)
        parts = next(reader, [])
        return [p.strip() for p in parts]
    except Exception:
        # Если CSV парсер сломался, используем простой split
        return [p.strip() for p in line.split(delimiter)]


def is_header_row(parts: list) -> bool:
    """Проверяет, похожа ли строка на заголовок"""
    if not parts or len(parts) == 0:
        return False
    
    # Список известных заголовков на разных языках и вариантах
    standard_headers = {
        # Телефон
        'phone', 'телефон', 'мобильный', 'mobile', 'tel', 'telephone', 'phone_number', 'phonenumber',
        'номер телефона', 'тел', 'моб', 'ph', 'мобильный номер',
        # Email
        'email', 'почта', 'e-mail', 'email_address', 'emailaddress', 'e_mail', 'письмо', 'адрес почты',
        # Имя
        'name', 'имя', 'фио', 'ф.и.о', 'полное имя', 'first_name', 'firstname',
        'fname', 'given_name', 'givenname', 'forename',
        # Фамилия
        'last_name', 'lastname', 'фамилия', 'surname', 'lname', 'family_name', 'familyname',
        'отчество', 'second_name',
        # ID/UID
        'id', 'uid', 'user_id', 'userid', 'account_id', 'accountid', 'код', 'номер', 'num', 'code',
        'account', 'user', 'идентификатор', 'код клиента',
        # Пол
        'gender', 'пол', 'sex', 'гендер',
        # Место
        'city', 'город', 'location', 'место', 'населенный пункт', 'нп',
        'oblast', 'область', 'регион', 'region', 'state', 'province',
        # Адрес
        'address', 'адрес', 'street', 'улица', 'ул', 'home', 'дом',
        'квартира', 'кв', 'apartment', 'flat',
        # Дата
        'date', 'дата', 'birth_date', 'bday', 'birthday', 'date_of_birth', 'дата рождения', 'др',
        'born', 'день рождения', 'birth',
        # Другие
        'password', 'пароль', 'username', 'пользователь', 'age', 'возраст', 'country', 'страна',
    }
    
    # Проверяем каждую часть заголовка
    parts_lower = [p.lower().strip().replace('"', '').replace("'", '') for p in parts]
    
    matches = 0
    for part in parts_lower:
        if not part:  # Пустая колонка
            continue
        
        # Точное совпадение
        if part in standard_headers:
            matches += 1
        else:
            # Проверяем вхождение (для "Phone Number", "First Name" и т.д.)
            for header in standard_headers:
                if header in part or part in header:
                    matches += 1
                    break
    
    # Если более 30% колонок - известные заголовки, это заголовок
    if len(parts_lower) > 0:
        ratio = matches / len(parts_lower)
        return ratio >= 0.3
    
    return False


def has_header(file_path: Path, encoding: str, delimiter: str) -> bool:
    """Определяет есть ли заголовок в CSV"""
    try:
        with file_path.open("r", encoding=encoding, errors="replace") as f:
            first_line = f.readline().rstrip("\n\r")
            second_line = f.readline().rstrip("\n\r")
            third_line = f.readline().rstrip("\n\r")
            
            if not first_line or not second_line:
                return False
            
            if first_line.startswith("\ufeff"):
                first_line = first_line.lstrip("\ufeff")
            if second_line.startswith("\ufeff"):
                second_line = second_line.lstrip("\ufeff")
            if third_line.startswith("\ufeff"):
                third_line = third_line.lstrip("\ufeff")
            
            # Парсим все три строки правильно с помощью CSV парсера
            first_parts = parse_csv_line(first_line, delimiter)
            second_parts = parse_csv_line(second_line, delimiter)
            third_parts = parse_csv_line(third_line, delimiter) if third_line else []
            
            # Проверка 1: похожа ли первая строка на заголовок
            if is_header_row(first_parts):
                return True
            
            # Проверка 2: если вторая и третья строки имеют больше колонок чем первая
            # значит первая - заголовок (потому что адреса в кавычках)
            if len(second_parts) > len(first_parts) and len(third_parts) > len(first_parts):
                # Но только если первая строка содержит слова
                text_count_first = sum(1 for p in first_parts if any(c.isalpha() for c in p))
                if text_count_first > len(first_parts) * 0.3:
                    return True
            
            # Проверка 3: если первая строка - это названия, а вторая - данные
            text_count_first = sum(1 for p in first_parts if any(c.isalpha() for c in p))
            text_count_second = sum(1 for p in second_parts if any(c.isdigit() for c in p))
            
            if text_count_first > len(first_parts) * 0.5 and text_count_second > len(second_parts) * 0.5:
                return True
            
            return False
    except Exception as e:
        logger.debug(f"Ошибка при определении заголовка: {e}")
        return False


def get_column_mapping(headers: list) -> dict:
    """Создает маппинг стандартных названий колонок"""
    mapping = {}
    headers_lower = [h.lower().strip().replace('"', '').replace("'", '') for h in headers]
    
    field_aliases = {
        'phone': [
            'phone', 'телефон', 'мобильный', 'mobile', 'tel', 'telephone', 'phone_number', 
            'phonenumber', 'номер телефона', 'тел', 'моб', 'ph',
        ],
        'email': [
            'email', 'почта', 'e-mail', 'email_address', 'emailaddress', 'work', 'home',
            'e_mail', 'письмо', 'адрес почты',
        ],
        'first_name': [
            'first_name', 'firstname', 'имя', 'fname', 'given_name', 'givenname', 'forename',
            'первое имя', 'name', 'first',
        ],
        'last_name': [
            'last_name', 'lastname', 'фамилия', 'surname', 'lname', 'family_name', 'familyname',
            'second_name', 'второе имя',
        ],
        'full_name': [
            'full_name', 'fullname', 'полное имя', 'name', 'nimi', 'person - name',
            'fio', 'ф.и.о', 'фио',
        ],
        'uid': [
            'uid', 'id', 'user_id', 'userid', 'account_id', 'accountid', 'код', 'номер',
            'code', 'account', 'user', 'идентификатор', 'код клиента',
        ],
        'gender': [
            'gender', 'пол', 'sex', 'гендер',
        ],
        'location': [
            'location', 'место', 'город', 'city', 'населенный пункт', 'нп',
        ],
        'oblast': [
            'oblast', 'область', 'регион', 'region', 'state', 'province',
        ],
        'city': [
            'city', 'город', 'населенный пункт',
        ],
        'street': [
            'street', 'улица', 'ул', 'street_address',
        ],
        'home': [
            'home', 'дом', 'house', 'home_number', 'house_number',
        ],
        'kvartita': [
            'kvartita', 'квартира', 'кв', 'apartment', 'flat', 'apt',
        ],
        'hometown': [
            'hometown', 'родной город', 'home_town', 'birth_city',
        ],
        'birth_date': [
            'birth_date', 'birthday', 'date_of_birth', 'дата рождения', 'bday', 'dr',
            'born', 'день рождения', 'birth', 'dob',
        ],
        'created_at': [
            'created_at', 'date_registered', 'registration_date', 'join_date', 'created',
        ],
        'name': [
            'name', 'полное имя', 'фио', 'ф.и.о', 'full_name', 'fullname',
        ],
        'bday': [
            'bday', 'birthday', 'date_of_birth', 'дата рождения', 'birth_date',
            'dr', 'born', 'день рождения', 'birth', 'dob',
        ],
        'code': [
            'code', 'код', 'номер', 'id', 'uid', 'код клиента',
        ],
    }
    
    for standard_name, aliases in field_aliases.items():
        for i, header in enumerate(headers_lower):
            found = False
            # Точное совпадение
            if header in aliases:
                mapping[standard_name] = i
                found = True
            else:
                # Проверяем вхождение для частичного совпадения
                for alias in aliases:
                    if alias in header or header in alias:
                        mapping[standard_name] = i
                        found = True
                        break
            if found:
                break
    
    return mapping


def index_csv(csv_path: Path):
    """Индексирует CSV файл с оптимизированной FTS5"""
    if not csv_path.is_file():
        logger.error(f"Файл не найден: {csv_path}")
        return

    base_name = csv_path.stem
    db_path = INDEX_DIR / f"{base_name}.db"
    metadata_path = INDEX_DIR / f"{base_name}.meta.json"
    offsets_path = INDEX_DIR / f"{base_name}.offsets.bin"

    try:
        file_size_mb = csv_path.stat().st_size / (1024**2)
        logger.info(f"→ Индексация {csv_path.name} ({file_size_mb:.1f} MiB)")

        encoding = detect_encoding(csv_path)
        delimiter = detect_delimiter(csv_path, encoding)
        has_header_row = has_header(csv_path, encoding, delimiter)
        
        logger.info(f"   encoding: {encoding} | delimiter: '{delimiter}' | header: {has_header_row}")

        # Удаляем старые файлы если они существуют
        for path in [db_path, offsets_path, metadata_path]:
            if path.exists():
                path.unlink(missing_ok=True)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -200000;
            PRAGMA temp_store = MEMORY;
        """)

        # Создаем ОПТИМИЗИРОВАННУЮ FTS5 таблицу - только нужные поля
        cur.execute("""
            DROP TABLE IF EXISTS records;
        """)
        
        # Индексируем только критические поля
        fields_sql = ", ".join(INDEXED_FIELDS)
        cur.execute(f"""
            CREATE VIRTUAL TABLE records USING fts5(
                {fields_sql},
                content=''
            );
        """)
        conn.commit()

        # Читаем первую строку для определения заголовков
        headers = []
        col_mapping = {}
        
        with csv_path.open("r", encoding=encoding, errors="replace") as f:
            first_line = f.readline().rstrip("\n\r")
            if first_line.startswith("\ufeff"):
                first_line = first_line.lstrip("\ufeff")
            
            first_parts = parse_csv_line(first_line, delimiter)
            
            if has_header_row:
                headers = first_parts
                col_mapping = get_column_mapping(headers)
                logger.info(f"   Колонки: {', '.join(headers[:10])}{'...' if len(headers) > 10 else ''}")
                logger.info(f"   Маппинг: {col_mapping}")
            else:
                # Если заголовка нет, пробуем определить по первой строке
                headers = [f"col_{i}" for i in range(len(first_parts))]
                logger.info(f"   Заголовков не найдено, используются позиции: {len(headers)} колонок")

        batch_data = []
        offsets = []
        position = 0
        processed_lines = 0
        skipped_lines = 0

        with csv_path.open("rb") as f:
            # Пропускаем заголовок если он есть
            if has_header_row:
                f.readline()
            position = f.tell()

            # Читаем данные
            for raw_line_bytes in f:
                # Сохраняем offset ДО чтения строки
                offsets.append(position)
                
                line_text = raw_line_bytes.decode(encoding, errors="replace").rstrip("\n\r")
                
                if line_text.startswith("\ufeff"):
                    line_text = line_text.lstrip("\ufeff")

                if not line_text.strip():
                    position = f.tell()
                    skipped_lines += 1
                    continue

                # Очищаем нулевые символы
                line_text = line_text.replace('\x00', '').strip()

                if line_text:
                    # Парсим строку правильно с помощью CSV парсера
                    parts = parse_csv_line(line_text, delimiter)
                    
                    # Формируем кортеж значений для нужных полей
                    values = []
                    for field in INDEXED_FIELDS:
                        col_idx = col_mapping.get(field, -1)
                        if col_idx >= 0 and col_idx < len(parts):
                            val = parts[col_idx].strip().replace('"', '')
                            values.append(val if val else None)
                        else:
                            values.append(None)
                    
                    # Вставляем только если есть хотя бы одно значение
                    if any(values):
                        batch_data.append(tuple(values))
                        processed_lines += 1

                        if len(batch_data) >= BATCH_SIZE:
                            placeholders = ", ".join(["?"] * len(INDEXED_FIELDS))
                            cur.executemany(
                                f"INSERT INTO records({fields_sql}) VALUES ({placeholders})",
                                batch_data
                            )
                            batch_data.clear()
                            conn.commit()

                        if processed_lines % COMMIT_EVERY == 0:
                            conn.commit()
                            logger.info(f"   commit на строке {processed_lines:,}")

                        if processed_lines % PROGRESS_EVERY == 0:
                            logger.info(f"   обработано строк: {processed_lines:,}")
                    else:
                        skipped_lines += 1

                position = f.tell()

        # Вставляем оставшиеся записи
        if batch_data:
            placeholders = ", ".join(["?"] * len(INDEXED_FIELDS))
            cur.executemany(
                f"INSERT INTO records({fields_sql}) VALUES ({placeholders})",
                batch_data
            )
            conn.commit()

        # Сохраняем offsets
        if offsets:
            with offsets_path.open("wb") as off_f:
                for off in offsets:
                    off_f.write(struct.pack("Q", off))

        # Сохраняем метаданные
        metadata = {
            "filename": csv_path.name,
            "encoding": encoding,
            "delimiter": delimiter,
            "has_header": has_header_row,
            "total_rows": processed_lines,
            "headers": headers,
            "column_mapping": {k: headers[v] if v < len(headers) else None for k, v in col_mapping.items()},
            "column_indices": col_mapping,
            "indexed_fields": INDEXED_FIELDS,
            "index_type": "optimized_fts5"
        }
        with metadata_path.open("w", encoding="utf-8") as mf:
            json.dump(metadata, mf, ensure_ascii=False, indent=2)

        # Оптимизируем БД
        try:
            logger.info("   VACUUM...")
            cur.execute("VACUUM;")
            conn.commit()
        except Exception as e:
            logger.debug(f"Ошибка при оптимизации: {e}")

        conn.close()

        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024**2)
            csv_size_mb = csv_path.stat().st_size / (1024**2)
            reduction = (1 - size_mb / csv_size_mb) * 100 if csv_size_mb > 0 else 0
            logger.info(f"✓ Готово | строк: {processed_lines:,} | пропущено: {skipped_lines:,} | индекс: {size_mb:.1f} MiB (-{reduction:.1f}%)")
        else:
            logger.error(f"✗ Файл БД не создан")

    except Exception as e:
        logger.exception(f"Ошибка при обработке {csv_path.name}")
        for path in [db_path, offsets_path, metadata_path]:
            if path.exists():
                path.unlink(missing_ok=True)


def main():
    csv_files = sorted(f for f in CSV_DIR.iterdir() if f.suffix.lower() == ".csv")
    if not csv_files:
        logger.info("CSV-файлы не найдены в " + str(CSV_DIR))
        return

    logger.info(f"Найдено CSV: {len(csv_files)}")

    for csv_file in csv_files:
        index_csv(csv_file)

    logger.info("Индексация завершена")


if __name__ == "__main__":
    main()
