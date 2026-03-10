import csv
import json
import os


def split_csv_arg(value):
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip() != ""]


def load_records(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of objects.")
        return data
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_records(path, records):
    records = [] if records is None else records
    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return
    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    fieldnames = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def resolve_path(path, base_path="", metadata_path=None):
    if path is None or path == "":
        return path
    if os.path.isabs(path):
        return path
    candidates = []
    if base_path not in (None, ""):
        candidates.append(os.path.join(base_path, path))
    if metadata_path not in (None, ""):
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(metadata_path)), path))
    candidates.append(os.path.abspath(path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def normalize_optional_str(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return None
    return value


def parse_json_like(value):
    value = normalize_optional_str(value)
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def try_float(value):
    value = normalize_optional_str(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
