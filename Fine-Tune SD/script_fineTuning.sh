export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="cartoon_inp"
export HUB_MODEL_ID="dlcvproj/inpaint_cartoon"
export DATASET_NAME="Norod78/cartoon-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=128 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub\
  --hub_model_id=${HUB_MODEL_ID} \
  --hub_token="" \
  --num_train_epochs=3 \
  --checkpointing_steps=250 \
  --validation_prompt="A happy man in a garden" \
  --seed=1337