# Server-based scoring through an OpenAI-compatible TextPecker endpoint.
# If you have a local TextPecker checkout, add:
#   --official_textpecker_root /path/to/TextPecker
python examples/z_image/model_training/special/text_preference/score_with_textpecker.py \
  --input_path data/text_preference_dataset/candidates.csv \
  --output_path data/text_preference_dataset/candidates_scored.csv \
  --base_path data/text_preference_dataset \
  --image_key image \
  --target_key target_text \
  --server_url http://127.0.0.1:8848/v1 \
  --model TextPecker \
  --api_key EMPTY \
  --workers 4 \
  --image_transport path
