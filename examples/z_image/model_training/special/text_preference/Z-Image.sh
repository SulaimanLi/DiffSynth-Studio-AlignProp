#
# Offline preference metadata is expected to contain at least:
# - prompt
# - chosen_image
# - rejected_image
#
# TextPecker scoring/filtering happens before this training script.
#
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
