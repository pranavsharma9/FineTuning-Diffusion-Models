export DATASET_NAME="Norod78/cartoon-blip-captions"

accelerate launch train_text_to_image_lora_prior.py \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=32 \
  --train_batch_size=1 \
  --max_train_steps=350 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --rank=4 \
  --report_to='wandb' \
  --validation_prompt="a man in a garden" \
  --push_to_hub \
  --dataloader_num_workers=0 \
  --output_dir="wuerstchen-prior-cartoon-lora"