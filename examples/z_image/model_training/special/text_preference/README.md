# Z-Image Text Preference Training

This directory contains a minimal offline workflow for:

1. scoring candidate images with TextPecker
2. converting scored candidates into `chosen/rejected` preference pairs
3. training Z-Image LoRA with the `text_pref_dpo` task

The recommended setup is:

* run TextPecker scoring inside the TextPecker / ms-swift environment
* run metadata building and Z-Image training inside the DiffSynth environment

## Files

* `score_with_textpecker_local.py`: local offline TextPecker scoring through Python import
* `TextPecker-Score-Local.sh`: example wrapper for local offline scoring
* `score_with_textpecker.py`: server-based scorer through an OpenAI-compatible endpoint
* `TextPecker-Score.sh`: example wrapper for the server-based scorer
* `build_preference_metadata.py`: convert scored candidates into `chosen/rejected` metadata
* `Build-Preference-Metadata.sh`: example wrapper for metadata conversion
* `Z-Image.sh`: example Z-Image preference training script

## Recommended Workflow

### 1. Prepare candidate metadata

Prepare a metadata file containing candidate images for the same prompt/text target. `csv`, `json`, and `jsonl` are all supported.

Minimal fields:

* `prompt`: generation prompt
* `target_text`: expected text content
* `image`: candidate image path

Optional fields:

* `box`: text box or region annotation used by TextPecker
* `group_id`: strongly recommended when one `prompt,target_text` pair has multiple independent candidate sets

Example `csv`:

```csv
group_id,prompt,target_text,image,box
sample_0001,"A clean poster with Chinese text","CN_HELLO_WORLD","candidates/sample_0001_0.png","[120,80,420,180]"
sample_0001,"A clean poster with Chinese text","CN_HELLO_WORLD","candidates/sample_0001_1.png","[120,80,420,180]"
sample_0001,"A clean poster with Chinese text","CN_HELLO_WORLD","candidates/sample_0001_2.png","[120,80,420,180]"
sample_0002,"A coffee ad with Chinese text","CN_TODAY_SPECIAL","candidates/sample_0002_0.png","[90,60,360,150]"
sample_0002,"A coffee ad with Chinese text","CN_TODAY_SPECIAL","candidates/sample_0002_1.png","[90,60,360,150]"
```

If you do not have `group_id`, the metadata builder will group by `prompt,target_text` by default. That is safe only when each prompt/text pair maps to one candidate set.

### 2. Score candidates with local TextPecker inference

Run this step inside the TextPecker / ms-swift environment.

Example:

```bash
python examples/z_image/model_training/special/text_preference/score_with_textpecker_local.py \
  --input_path data/text_preference_dataset/candidates.csv \
  --output_path data/text_preference_dataset/candidates_scored.csv \
  --base_path data/text_preference_dataset \
  --image_key image \
  --target_key target_text \
  --box_key box \
  --model /path/to/TextPecker-model \
  --batch_size 1 \
  --request_format auto \
  --official_textpecker_root /path/to/TextPecker
```

Important arguments:

* `--model`: local TextPecker model path or model id used by Swift
* `--official_textpecker_root`: local TextPecker repo root. Use this when you want score parsing to stay aligned with official `parse_utils_pecker`
* `--request_format auto`: first tries OpenAI-style multimodal `InferRequest`, then falls back to legacy `"<image>prompt"` style
* `--box_key`: optional field for region-aware scoring
* `--batch_size`: keep this conservative first; start from `1`

The scorer tries two layers:

* official parser: imports `parse_utils_pecker`
* fallback parser: if official parser is unavailable, it extracts `recognized_text` and estimates quality/semantic scores approximately

The scored output appends fields such as:

* `textpecker_recognized_text`
* `textpecker_quality_score`
* `textpecker_semantic_score`
* `textpecker_combined_score`

If a row fails, it will contain `textpecker_error`.

### 3. Build preference metadata

After scoring, convert the scored candidates into `chosen/rejected` pairs.

Example:

```bash
python examples/z_image/model_training/special/text_preference/build_preference_metadata.py \
  --input_path data/text_preference_dataset/candidates_scored.csv \
  --output_path data/text_preference_dataset/metadata.csv \
  --image_key image \
  --score_key textpecker_combined_score \
  --semantic_score_key textpecker_semantic_score \
  --quality_score_key textpecker_quality_score \
  --recognized_text_key textpecker_recognized_text \
  --group_keys group_id \
  --copy_keys prompt,target_text,group_id \
  --min_group_size 2 \
  --min_score_gap 0.05
```

Recommended grouping:

* if you have a stable candidate-set id, use `--group_keys group_id`
* otherwise use `--group_keys prompt,target_text`

The output contains:

* `prompt`
* `target_text`
* `chosen_image`
* `rejected_image`
* `chosen_score`
* `rejected_score`
* `score_gap`
* optional recognized-text and sub-score fields

### 4. Train Z-Image preference LoRA

Run this step inside the DiffSynth environment.

Example:

```bash
accelerate launch examples/z_image/model_training/train.py \
  --dataset_base_path data/text_preference_dataset \
  --dataset_metadata_path data/text_preference_dataset/metadata.csv \
  --data_file_keys "chosen_image,rejected_image" \
  --max_pixels 1048576 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Z-Image_text_pref_dpo_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --task "text_pref_dpo" \
  --preference_loss_type "dpo" \
  --preference_beta 100 \
  --preference_num_timesteps 4 \
  --preference_loss_weight 1.0 \
  --preference_sft_weight 0.1
```

## Training Task Summary

`text_pref_dpo` is a flow-matching preference objective with optional SFT regularization.

Key arguments:

* `--preference_loss_type`: `dpo` or `ranking`
* `--preference_beta`: inverse temperature for pairwise preference loss
* `--preference_num_timesteps`: number of sampled flow-matching timesteps per comparison
* `--preference_loss_weight`: weight of preference loss
* `--preference_sft_weight`: weight of chosen-sample SFT regularization

Notes:

* `dpo` uses a frozen reference copy of the current iteration models
* `ranking` does not use a reference model and is cheaper
* split-cache training for `text_pref_dpo` is not implemented in the current version

## Environment Split

Recommended environment split:

* TextPecker environment:
  * `score_with_textpecker_local.py`
* DiffSynth environment:
  * `build_preference_metadata.py`
  * `train.py` with `--task text_pref_dpo`

You can also run `build_preference_metadata.py` in the TextPecker environment if plain Python dependencies are available.

## Server-Based Scoring

`score_with_textpecker.py` is the server-based alternative. It sends requests to a deployed OpenAI-compatible endpoint such as:

```text
http://127.0.0.1:8848/v1/chat/completions
```

Use this only when your environment allows serving a local or remote endpoint. In restricted environments, prefer `score_with_textpecker_local.py`.

## Common Issues

### Scores look inconsistent with official TextPecker

Pass `--official_textpecker_root /path/to/TextPecker` so the script imports official `parse_utils_pecker`.

### Different candidate sets were mixed together

Add a dedicated grouping field such as `group_id`, then use:

```bash
--group_keys group_id
```

### Local inference request format fails

Keep `--request_format auto`. If needed, force one of:

```bash
--request_format openai
--request_format legacy
```

### There are many `textpecker_error` rows

Check:

* the image paths resolved by `--base_path`
* whether the TextPecker model path is correct
* whether the current environment can import `swift`
* whether the official parser path is valid when `--official_textpecker_root` is used

## Quick Start

From repo root:

```bash
# Step 1: run inside the TextPecker environment
python examples/z_image/model_training/special/text_preference/score_with_textpecker_local.py \
  --input_path data/text_preference_dataset/candidates.csv \
  --output_path data/text_preference_dataset/candidates_scored.csv \
  --base_path data/text_preference_dataset \
  --image_key image \
  --target_key target_text \
  --model /path/to/TextPecker-model \
  --official_textpecker_root /path/to/TextPecker

# Step 2: run inside the DiffSynth environment
python examples/z_image/model_training/special/text_preference/build_preference_metadata.py \
  --input_path data/text_preference_dataset/candidates_scored.csv \
  --output_path data/text_preference_dataset/metadata.csv \
  --image_key image \
  --score_key textpecker_combined_score \
  --group_keys group_id \
  --copy_keys prompt,target_text,group_id \
  --min_score_gap 0.05

# Step 3: run inside the DiffSynth environment
accelerate launch examples/z_image/model_training/train.py \
  --dataset_base_path data/text_preference_dataset \
  --dataset_metadata_path data/text_preference_dataset/metadata.csv \
  --data_file_keys "chosen_image,rejected_image" \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --output_path "./models/train/Z-Image_text_pref_dpo_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --task "text_pref_dpo"
```
