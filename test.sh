# 1. 参数区
CKPT="output/stylegan_full/last_epoch_100.pth"   # checkpoint 路径
MODEL="stylegan"                                 # 与 checkpoint 匹配
NUM=16                                           # 生成图片数量
NROW=4                                           # 每行放几个
OUT="test_result.jpg"

echo "[INFO] Testing $MODEL from $CKPT"

# 2. Python 命令
python3 -u src/test.py \
    --ckpt "$CKPT" \
    --num "$NUM" \
    --model "$MODEL" \
    --nrow "$NROW" \
    --out "$OUT"

echo "[INFO] Test image saved to $OUT"
