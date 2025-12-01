# 1. 可调参数
MODEL="lsgan"          # 选项: dcgan / lsgan / wgangp / stylegan
EPOCHS=100
BATCH=64
IMAGE_SIZE=64
Z_DIM=128
LR_G=0.0002
LR_D=0.00003

echo "[INFO] Training $MODEL"
echo "[INFO] Epochs: $EPOCHS, Batch: $BATCH, z_dim: $Z_DIM"

# 2. Python 启动命令
python3 -u src/main.py \
    --model_name "$MODEL" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --image_size "$IMAGE_SIZE" \
    --z_dim "$Z_DIM" \
    --lr_G "$LR_G" \
    --lr_D "$LR_D"

echo "[INFO] Training completed."
