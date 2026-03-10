import argparse
from collections import defaultdict

from common import load_records, normalize_optional_str, save_records, split_csv_arg, try_float


def parse_args():
    parser = argparse.ArgumentParser(description="Build chosen/rejected preference metadata from scored candidate images.")
    parser.add_argument("--input_path", type=str, required=True, help="Scored candidate metadata path.")
    parser.add_argument("--output_path", type=str, required=True, help="Output preference metadata path.")
    parser.add_argument("--image_key", type=str, default="image", help="Image field in scored metadata.")
    parser.add_argument("--score_key", type=str, default="textpecker_combined_score", help="Score field used to rank candidates.")
    parser.add_argument("--semantic_score_key", type=str, default="textpecker_semantic_score", help="Optional semantic score field used as tie-breaker and copied to output.")
    parser.add_argument("--quality_score_key", type=str, default="textpecker_quality_score", help="Optional quality score field used as tie-breaker and copied to output.")
    parser.add_argument("--recognized_text_key", type=str, default="textpecker_recognized_text", help="Optional recognized text field copied to output.")
    parser.add_argument("--group_keys", type=str, default="prompt,target_text", help="Comma-separated keys used to group candidates into one preference set.")
    parser.add_argument("--copy_keys", type=str, default="prompt,target_text", help="Comma-separated keys copied from the best candidate row.")
    parser.add_argument("--min_group_size", type=int, default=2, help="Minimum valid candidates required in one group.")
    parser.add_argument("--min_score_gap", type=float, default=0.05, help="Minimum score gap required between chosen and rejected.")
    parser.add_argument("--min_chosen_score", type=float, default=None, help="Optional lower bound for the chosen score.")
    parser.add_argument("--max_rejected_score", type=float, default=None, help="Optional upper bound for the rejected score.")
    parser.add_argument("--chosen_image_key", type=str, default="chosen_image", help="Output field name for the chosen image.")
    parser.add_argument("--rejected_image_key", type=str, default="rejected_image", help="Output field name for the rejected image.")
    return parser.parse_args()


def build_group_id(row, group_keys):
    return tuple(normalize_optional_str(row.get(key)) or "" for key in group_keys)


def rank_candidates(rows, primary_score_key, semantic_score_key=None, quality_score_key=None):
    def sort_key(row):
        primary = try_float(row.get(primary_score_key))
        semantic = try_float(row.get(semantic_score_key)) if semantic_score_key is not None else None
        quality = try_float(row.get(quality_score_key)) if quality_score_key is not None else None
        return (
            float("-inf") if primary is None else primary,
            float("-inf") if semantic is None else semantic,
            float("-inf") if quality is None else quality,
        )

    return sorted(rows, key=sort_key, reverse=True)


def main():
    args = parse_args()
    group_keys = split_csv_arg(args.group_keys)
    copy_keys = split_csv_arg(args.copy_keys)
    if len(group_keys) == 0:
        raise ValueError("group_keys must not be empty.")

    records = load_records(args.input_path)
    grouped_records = defaultdict(list)
    for record in records:
        score = try_float(record.get(args.score_key))
        if score is None:
            continue
        grouped_records[build_group_id(record, group_keys)].append(record)

    preference_records = []
    skipped_groups = 0
    for _, rows in grouped_records.items():
        if len(rows) < args.min_group_size:
            skipped_groups += 1
            continue
        ranked_rows = rank_candidates(rows, args.score_key, args.semantic_score_key, args.quality_score_key)
        chosen = ranked_rows[0]
        rejected = ranked_rows[-1]

        chosen_score = try_float(chosen.get(args.score_key))
        rejected_score = try_float(rejected.get(args.score_key))
        if chosen_score is None or rejected_score is None:
            skipped_groups += 1
            continue
        if args.min_chosen_score is not None and chosen_score < args.min_chosen_score:
            skipped_groups += 1
            continue
        if args.max_rejected_score is not None and rejected_score > args.max_rejected_score:
            skipped_groups += 1
            continue
        if chosen_score - rejected_score < args.min_score_gap:
            skipped_groups += 1
            continue
        chosen_image = normalize_optional_str(chosen.get(args.image_key))
        rejected_image = normalize_optional_str(rejected.get(args.image_key))
        if chosen_image is None or rejected_image is None or chosen_image == rejected_image:
            skipped_groups += 1
            continue

        preference_record = {}
        for key in copy_keys:
            if key in chosen:
                preference_record[key] = chosen[key]
        preference_record[args.chosen_image_key] = chosen_image
        preference_record[args.rejected_image_key] = rejected_image
        preference_record["chosen_score"] = chosen_score
        preference_record["rejected_score"] = rejected_score
        preference_record["score_gap"] = chosen_score - rejected_score
        if args.recognized_text_key in chosen:
            preference_record["chosen_recognized_text"] = chosen[args.recognized_text_key]
        if args.recognized_text_key in rejected:
            preference_record["rejected_recognized_text"] = rejected[args.recognized_text_key]
        if args.semantic_score_key in chosen:
            preference_record["chosen_semantic_score"] = chosen[args.semantic_score_key]
        if args.semantic_score_key in rejected:
            preference_record["rejected_semantic_score"] = rejected[args.semantic_score_key]
        if args.quality_score_key in chosen:
            preference_record["chosen_quality_score"] = chosen[args.quality_score_key]
        if args.quality_score_key in rejected:
            preference_record["rejected_quality_score"] = rejected[args.quality_score_key]
        preference_records.append(preference_record)

    save_records(args.output_path, preference_records)
    print(f"Saved {len(preference_records)} preference pairs to: {args.output_path}")
    print(f"Skipped groups: {skipped_groups}")


if __name__ == "__main__":
    main()
