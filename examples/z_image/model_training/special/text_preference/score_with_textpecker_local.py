import argparse
import inspect
import json
import os
import traceback

from common import load_records, normalize_optional_str, parse_json_like, resolve_path, save_records, split_csv_arg, try_float
from score_with_textpecker import (
    OfficialTextPeckerScorer,
    aggregate_score,
    build_prompt,
    fallback_score,
    parse_message_content,
)


def import_swift_runtime():
    try:
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

        return {
            "engine_class": TransformersEngine,
            "infer_request_class": InferRequest,
            "request_config_class": RequestConfig,
            "variant": "swift4",
        }
    except ImportError:
        from swift.llm import InferRequest, PtEngine, RequestConfig

        return {
            "engine_class": PtEngine,
            "infer_request_class": InferRequest,
            "request_config_class": RequestConfig,
            "variant": "swift3",
        }


def build_engine(runtime, model, adapters=None, engine_kwargs=None):
    engine_kwargs = {} if engine_kwargs is None else dict(engine_kwargs)
    engine_class = runtime["engine_class"]
    signature = inspect.signature(engine_class)
    parameters = signature.parameters

    if adapters is not None and "adapters" in parameters:
        engine_kwargs["adapters"] = adapters

    if "model_id_or_path" in parameters:
        return engine_class(model_id_or_path=model, **engine_kwargs)
    if "model" in parameters:
        return engine_class(model=model, **engine_kwargs)
    return engine_class(model, **engine_kwargs)


def build_request_openai_format(runtime, prompt, image_path, system=None):
    infer_request_class = runtime["infer_request_class"]
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return infer_request_class(messages=messages)


def build_request_legacy_format(runtime, prompt, image_path, system=None):
    infer_request_class = runtime["infer_request_class"]
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": f"<image>{prompt}"})
    return infer_request_class(messages=messages, images=[image_path])


def build_requests(runtime, prompt_items, request_format="auto", system=None):
    if request_format in ("openai", "auto"):
        return [build_request_openai_format(runtime, item["prompt"], item["image_path"], system=system) for item in prompt_items], "openai"
    if request_format == "legacy":
        return [build_request_legacy_format(runtime, item["prompt"], item["image_path"], system=system) for item in prompt_items], "legacy"
    raise ValueError(f"Unsupported request_format: {request_format}")


def build_request_config(runtime, max_tokens, temperature):
    request_config_class = runtime["request_config_class"]
    signature = inspect.signature(request_config_class)
    kwargs = {}
    if "max_tokens" in signature.parameters:
        kwargs["max_tokens"] = max_tokens
    if "temperature" in signature.parameters:
        kwargs["temperature"] = temperature
    return request_config_class(**kwargs)


def extract_response_text(response):
    message = response.choices[0].message.content
    return parse_message_content(message)


def infer_batch(engine, runtime, prompt_items, request_config, request_format="auto", system=None):
    requests, active_format = build_requests(runtime, prompt_items, request_format=request_format, system=system)
    try:
        responses = engine.infer(requests, request_config=request_config)
        if not isinstance(responses, list):
            responses = [responses]
        return responses, active_format
    except Exception:
        if request_format != "auto":
            raise
    requests, active_format = build_requests(runtime, prompt_items, request_format="legacy", system=system)
    responses = engine.infer(requests, request_config=request_config)
    if not isinstance(responses, list):
        responses = [responses]
    return responses, active_format


def batch_iter(items, batch_size):
    batch_size = max(1, batch_size)
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-score images with local TextPecker inference through Swift Python APIs.")
    parser.add_argument("--input_path", type=str, required=True, help="Input metadata path. Supports csv/json/jsonl.")
    parser.add_argument("--output_path", type=str, required=True, help="Output metadata path. Supports csv/json/jsonl.")
    parser.add_argument("--base_path", type=str, default="", help="Base path used to resolve relative image paths.")
    parser.add_argument("--image_key", type=str, default="image", help="Image path field in the metadata.")
    parser.add_argument("--target_key", type=str, default="target_text", help="Ground-truth text field in the metadata.")
    parser.add_argument("--box_key", type=str, default=None, help="Optional text box field in the metadata.")
    parser.add_argument("--model", type=str, required=True, help="Local model path or model id used by Swift.")
    parser.add_argument("--adapters", type=str, default=None, help="Optional comma-separated adapter paths, if the Swift engine supports them.")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--request_format", type=str, default="auto", choices=("auto", "openai", "legacy"), help="InferRequest message format.")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    parser.add_argument("--official_textpecker_root", type=str, default=None, help="Optional local TextPecker repo root for importing parse_utils_pecker.")
    parser.add_argument("--output_prefix", type=str, default="textpecker", help="Prefix used for appended score columns.")
    parser.add_argument("--keep_raw_response", default=False, action="store_true", help="Whether to keep prompt/raw response in output metadata.")
    parser.add_argument("--attn_impl", type=str, default=None, help="Optional attn_impl passed to the Swift engine if supported.")
    parser.add_argument("--model_type", type=str, default=None, help="Optional model_type passed to the Swift engine if supported.")
    parser.add_argument("--template", type=str, default=None, help="Optional template passed to the Swift engine if supported.")
    parser.add_argument("--max_batch_size", type=int, default=None, help="Optional max_batch_size passed to the Swift engine if supported.")
    parser.add_argument("--engine_kwargs", type=str, default=None, help="Optional JSON dict merged into the engine constructor kwargs.")
    return parser.parse_args()


def load_official_scorer(repo_root=None):
    try:
        return OfficialTextPeckerScorer(repo_root)
    except Exception:
        if repo_root is not None:
            raise
        return None


def make_engine_kwargs(args, runtime):
    kwargs = {}
    engine_class = runtime["engine_class"]
    parameters = inspect.signature(engine_class).parameters
    if args.attn_impl is not None and "attn_impl" in parameters:
        kwargs["attn_impl"] = args.attn_impl
    if args.model_type is not None and "model_type" in parameters:
        kwargs["model_type"] = args.model_type
    if args.template is not None and "template" in parameters:
        kwargs["template"] = args.template
    if args.max_batch_size is not None and "max_batch_size" in parameters:
        kwargs["max_batch_size"] = args.max_batch_size
    if args.engine_kwargs is not None:
        extra_kwargs = json.loads(args.engine_kwargs)
        kwargs.update(extra_kwargs)
    return kwargs


def prepare_prompt_items(records, args, official_scorer):
    prompt_items = []
    prepared_rows = []
    for row in records:
        image_value = normalize_optional_str(row.get(args.image_key))
        if image_value is None:
            prepared_rows.append((row, None, "Missing image path"))
            continue
        target_text = normalize_optional_str(row.get(args.target_key)) if args.target_key is not None else None
        raw_box = parse_json_like(row.get(args.box_key)) if args.box_key is not None and args.box_key in row else None
        box = official_scorer.prepare_box(raw_box) if official_scorer is not None else raw_box
        resolved_image_path = resolve_path(image_value, base_path=args.base_path, metadata_path=args.input_path)
        prompt_items.append(
            {
                "row": row,
                "target_text": target_text,
                "raw_box": raw_box,
                "image_path": resolved_image_path,
                "prompt": build_prompt(target_text=target_text, box=box),
            }
        )
        prepared_rows.append((row, prompt_items[-1], None))
    return prepared_rows


def score_response(item, response_text, args, official_scorer):
    if official_scorer is not None:
        quality_score, semantic_score, recognized_text, full_results = official_scorer.score(
            response_text=response_text,
            target_text=item["target_text"],
            raw_box=item["raw_box"],
        )
    else:
        quality_score, semantic_score, recognized_text, full_results = fallback_score(
            response_text=response_text,
            target_text=item["target_text"],
        )

    quality_score = try_float(quality_score)
    semantic_score = try_float(semantic_score)
    combined_score = aggregate_score(quality_score, semantic_score)

    result = dict(item["row"])
    result[f"{args.output_prefix}_resolved_image_path"] = item["image_path"]
    result[f"{args.output_prefix}_recognized_text"] = recognized_text
    result[f"{args.output_prefix}_quality_score"] = quality_score
    result[f"{args.output_prefix}_semantic_score"] = semantic_score
    result[f"{args.output_prefix}_combined_score"] = combined_score
    if args.keep_raw_response:
        result[f"{args.output_prefix}_prompt"] = item["prompt"]
        result[f"{args.output_prefix}_raw_response"] = response_text
        result[f"{args.output_prefix}_parsed_result"] = json.dumps(full_results, ensure_ascii=False)
    return result


def main():
    args = parse_args()
    runtime = import_swift_runtime()
    official_scorer = None
    try:
        official_scorer = load_official_scorer(args.official_textpecker_root)
        if official_scorer is not None:
            print("Loaded official TextPecker parser.")
        else:
            print("Official TextPecker parser not found. Fallback scorer will be used.")
    except Exception as e:
        print(f"Warning: failed to import official TextPecker parser. {e}")

    adapters = split_csv_arg(args.adapters)
    engine_kwargs = make_engine_kwargs(args, runtime)
    engine = build_engine(runtime, model=args.model, adapters=adapters or None, engine_kwargs=engine_kwargs)
    request_config = build_request_config(runtime, max_tokens=args.max_tokens, temperature=args.temperature)

    records = load_records(args.input_path)
    prepared_rows = prepare_prompt_items(records, args, official_scorer)
    results = []
    infer_items = [item for _, item, error in prepared_rows if item is not None and error is None]

    for start, batch in batch_iter(infer_items, args.batch_size):
        try:
            responses, active_format = infer_batch(
                engine,
                runtime,
                batch,
                request_config=request_config,
                request_format=args.request_format,
                system=args.system,
            )
            print(f"[{start + len(batch)}/{len(infer_items)}] inferred with format={active_format}")
            for item, response in zip(batch, responses):
                response_text = extract_response_text(response)
                results.append(score_response(item, response_text, args, official_scorer))
        except Exception as e:
            error_text = "".join(traceback.format_exception_only(type(e), e)).strip()
            print(f"Batch inference failed at batch starting {start}: {error_text}")
            for item in batch:
                row = dict(item["row"])
                row[f"{args.output_prefix}_resolved_image_path"] = item["image_path"]
                row[f"{args.output_prefix}_error"] = error_text
                if args.keep_raw_response:
                    row[f"{args.output_prefix}_prompt"] = item["prompt"]
                results.append(row)

    final_results = []
    consumed = 0
    result_iter = iter(results)
    for original_row, prepared_item, error in prepared_rows:
        if error is not None:
            row = dict(original_row)
            row[f"{args.output_prefix}_error"] = error
            final_results.append(row)
            continue
        final_results.append(next(result_iter))
        consumed += 1

    save_records(args.output_path, final_results)
    print(f"Saved scored metadata to: {args.output_path}")
    print(f"Scored rows: {consumed}, total rows: {len(final_results)}")


if __name__ == "__main__":
    main()
