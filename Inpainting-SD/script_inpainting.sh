export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR="cartoon_inpaint"
export OUTPUT_DIR="sd-new-inpaint-dir"
export HUB_MODEL_ID="dlcvproj/sd_retro_inpaint"

accelerate launch train_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_data_dir="cartoon_inpaint" \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="in style of <cmg> " \
  --class_prompt="a cartoon image"
  --resolution=128 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --push_to_hub\
  --checkpointing_steps=1000
  --hub_model_id=${HUB_MODEL_ID} \
  --hub_token="" \
  --lr_warmup_steps=0 \
  --num_train_epochs=100\

  