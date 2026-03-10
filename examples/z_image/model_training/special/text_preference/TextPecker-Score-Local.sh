# Run this script inside the TextPecker / ms-swift environment.
# If parse_utils_pecker is not importable from the environment,
# add: --official_textpecker_root /path/to/TextPecker
python examples/z_image/model_training/special/text_preference/score_with_textpecker_local.py \
  --input_path data/text_preference_dataset/candidates.csv \
  --output_path data/text_preference_dataset/candidates_scored.csv \
  --base_path data/text_preference_dataset \
  --image_key image \
  --target_key target_text \
  --model /path/to/TextPecker-model \
  --batch_size 1 \
  --request_format auto
