python examples/z_image/model_training/special/text_preference/build_preference_metadata.py \
  --input_path data/text_preference_dataset/candidates_scored.csv \
  --output_path data/text_preference_dataset/metadata.csv \
  --image_key image \
  --score_key textpecker_combined_score \
  --group_keys prompt,target_text \
  --copy_keys prompt,target_text \
  --min_group_size 2 \
  --min_score_gap 0.05
