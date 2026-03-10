import argparse
import base64
import importlib
import json
import mimetypes
import os
import re
import sys
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from common import load_records, normalize_optional_str, parse_json_like, resolve_path, save_records, try_float


MARKER_PATTERN = re.compile(r"<###>|<#>")


def build_prompt(target_text=None, box=None):
    if target_text is not None and box is not None:
        return (
            "Please recognize the text in the image and assess its rendering quality. "
            f"The target text is: {target_text}. "
            f"The target text box is at {box}. "
            "Use `<###>` to mark illegible spans and return JSON with key `recognized_text`."
        )
    if target_text is not None:
        return (
            "Please recognize the text in the image and assess its rendering quality. "
            f"The target text is: {target_text}. "
            "Use `<###>` to mark illegible spans and return JSON with key `recognized_text`."
        )
    return (
        "Please recognize the text in the image and assess its rendering quality. "
        "Use `<###>` to mark illegible spans and return JSON with key `recognized_text`."
    )


def parse_message_content(message_content):
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts = []
        for item in message_content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "output_text" and "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(message_content)


def load_json_object_from_text(text):
    text = normalize_optional_str(text)
    if text is None:
        return None
    candidates = [text]
    code_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates.extend(code_blocks)
    object_candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    candidates.extend(object_candidates)
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate == "":
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def clean_recognized_text(text):
    text = normalize_optional_str(text)
    if text is None:
        return ""
    text = MARKER_PATTERN.sub("", text)
    text = re.sub(r"\s+", "", text)
    return text


def estimate_quality_score(recognized_text):
    recognized_text = normalize_optional_str(recognized_text)
    if recognized_text is None:
        return 0.0
    marker_count = len(MARKER_PATTERN.findall(recognized_text))
    clean_text = clean_recognized_text(recognized_text)
    content_units = len(clean_text)
    total_units = content_units + marker_count
    if total_units == 0:
        return 0.0
    return max(0.0, min(1.0, content_units / total_units))


def estimate_semantic_score(recognized_text, target_text):
    target_text = normalize_optional_str(target_text)
    if target_text is None:
        return None
    pred = clean_recognized_text(recognized_text)
    target = clean_recognized_text(target_text)
    if pred == "" and target == "":
        return 1.0
    if pred == "" or target == "":
        return 0.0
    return max(0.0, min(1.0, SequenceMatcher(None, pred, target).ratio()))


def aggregate_score(*scores):
    numeric_scores = [float(score) for score in scores if score is not None]
    if len(numeric_scores) == 0:
        return None
    return sum(numeric_scores) / len(numeric_scores)


def make_image_url(image_path, image_transport="path"):
    if image_transport == "path":
        return image_path
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


class OpenAICompatibleClient:
    def __init__(self, base_url, model, api_key="EMPTY", timeout=120, repetition_penalty=1.2, image_transport="path"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.repetition_penalty = repetition_penalty
        self.image_transport = image_transport

    def chat(self, prompt, image_path, max_tokens=2048):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": make_image_url(image_path, self.image_transport)}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response_json = json.loads(response.read().decode("utf-8"))
        message = response_json["choices"][0]["message"]["content"]
        return parse_message_content(message), response_json


class OfficialTextPeckerScorer:
    def __init__(self, repo_root=None):
        if repo_root is not None:
            repo_root = os.path.abspath(repo_root)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
        self.module = importlib.import_module("parse_utils_pecker")
        self.get_score_v2 = getattr(self.module, "get_score_v2", None)
        self.process_box_input = getattr(self.module, "process_box_input", None)
        if self.get_score_v2 is None:
            raise ImportError("parse_utils_pecker.get_score_v2 was not found.")

    def prepare_box(self, box):
        if box is None:
            return None
        if self.process_box_input is None:
            return box
        return self.process_box_input(box)

    def score(self, response_text, target_text=None, raw_box=None):
        message_raw = None
        if raw_box is not None or target_text is not None:
            message_raw = {"raw_box": raw_box, "raw_input": target_text}
        quality_score, semantic_score, full_results = self.get_score_v2(
            response_text,
            gt=target_text,
            message_raw=message_raw,
            key="all_info",
        )
        recognized_text = None
        if isinstance(full_results, dict):
            recognized_text = full_results.get("recognized_text")
        return quality_score, semantic_score, recognized_text, full_results


def fallback_score(response_text, target_text=None):
    parsed = load_json_object_from_text(response_text)
    recognized_text = None
    if isinstance(parsed, dict):
        recognized_text = parsed.get("recognized_text")
    if recognized_text is None:
        recognized_text = response_text
    quality_score = estimate_quality_score(recognized_text)
    semantic_score = estimate_semantic_score(recognized_text, target_text)
    full_results = parsed if parsed is not None else {"recognized_text": recognized_text}
    return quality_score, semantic_score, recognized_text, full_results


def score_record(
    row,
    input_path,
    client,
    image_key,
    target_key,
    box_key,
    base_path,
    official_scorer,
    max_tokens,
    prefix,
    keep_raw_response,
):
    image_value = normalize_optional_str(row.get(image_key))
    if image_value is None:
        raise ValueError(f"Missing image path in field `{image_key}`.")
    target_text = normalize_optional_str(row.get(target_key)) if target_key is not None else None
    raw_box = parse_json_like(row.get(box_key)) if box_key is not None and box_key in row else None
    box = official_scorer.prepare_box(raw_box) if official_scorer is not None else raw_box
    prompt = build_prompt(target_text=target_text, box=box)
    resolved_image_path = resolve_path(image_value, base_path=base_path, metadata_path=input_path)
    response_text, response_json = client.chat(prompt=prompt, image_path=resolved_image_path, max_tokens=max_tokens)

    if official_scorer is not None:
        quality_score, semantic_score, recognized_text, full_results = official_scorer.score(
            response_text=response_text,
            target_text=target_text,
            raw_box=raw_box,
        )
    else:
        quality_score, semantic_score, recognized_text, full_results = fallback_score(
            response_text=response_text,
            target_text=target_text,
        )

    quality_score = try_float(quality_score)
    semantic_score = try_float(semantic_score)
    combined_score = aggregate_score(quality_score, semantic_score)

    result = dict(row)
    result[f"{prefix}_resolved_image_path"] = resolved_image_path
    result[f"{prefix}_recognized_text"] = recognized_text
    result[f"{prefix}_quality_score"] = quality_score
    result[f"{prefix}_semantic_score"] = semantic_score
    result[f"{prefix}_combined_score"] = combined_score
    if keep_raw_response:
        result[f"{prefix}_prompt"] = prompt
        result[f"{prefix}_raw_response"] = response_text
        result[f"{prefix}_raw_json"] = json.dumps(response_json, ensure_ascii=False)
        result[f"{prefix}_parsed_result"] = json.dumps(full_results, ensure_ascii=False)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-score images with a deployed TextPecker endpoint.")
    parser.add_argument("--input_path", type=str, required=True, help="Input metadata file path. Supports csv/json/jsonl.")
    parser.add_argument("--output_path", type=str, required=True, help="Output metadata file path. Supports csv/json/jsonl.")
    parser.add_argument("--base_path", type=str, default="", help="Base path used to resolve relative image paths.")
    parser.add_argument("--image_key", type=str, default="image", help="Image path field in the metadata.")
    parser.add_argument("--target_key", type=str, default="target_text", help="Ground-truth text field in the metadata.")
    parser.add_argument("--box_key", type=str, default=None, help="Optional text box field in the metadata.")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8848/v1", help="TextPecker deployment URL.")
    parser.add_argument("--model", type=str, default="TextPecker", help="Model name used by the OpenAI-compatible endpoint.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key used by the endpoint.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent requests.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max output tokens.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty forwarded to the server.")
    parser.add_argument("--image_transport", type=str, default="path", choices=("path", "data_url"), help="How to send image content to the server.")
    parser.add_argument("--official_textpecker_root", type=str, default=None, help="Optional local TextPecker repo root for importing parse_utils_pecker.")
    parser.add_argument("--output_prefix", type=str, default="textpecker", help="Prefix used for appended score columns.")
    parser.add_argument("--keep_raw_response", default=False, action="store_true", help="Whether to keep prompt/raw response/raw json in output metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_records(args.input_path)
    client = OpenAICompatibleClient(
        base_url=args.server_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        repetition_penalty=args.repetition_penalty,
        image_transport=args.image_transport,
    )
    official_scorer = None
    if args.official_textpecker_root is not None:
        try:
            official_scorer = OfficialTextPeckerScorer(args.official_textpecker_root)
            print(f"Loaded official TextPecker parser from: {args.official_textpecker_root}")
        except Exception as e:
            print(f"Warning: failed to import official TextPecker parser, fallback scorer will be used. {e}")

    results = [None] * len(records)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                score_record,
                row=record,
                input_path=args.input_path,
                client=client,
                image_key=args.image_key,
                target_key=args.target_key,
                box_key=args.box_key,
                base_path=args.base_path,
                official_scorer=official_scorer,
                max_tokens=args.max_tokens,
                prefix=args.output_prefix,
                keep_raw_response=args.keep_raw_response,
            ): index
            for index, record in enumerate(records)
        }
        completed = 0
        for future in as_completed(future_map):
            index = future_map[future]
            row = dict(records[index])
            try:
                results[index] = future.result()
            except Exception as e:
                error_text = "".join(traceback.format_exception_only(type(e), e)).strip()
                if isinstance(e, urllib.error.HTTPError):
                    body = e.read().decode("utf-8", errors="ignore")
                    error_text = f"{error_text}: {body}"
                row[f"{args.output_prefix}_error"] = error_text
                results[index] = row
            completed += 1
            print(f"[{completed}/{len(records)}] scored")

    save_records(args.output_path, results)
    print(f"Saved scored metadata to: {args.output_path}")


if __name__ == "__main__":
    main()
