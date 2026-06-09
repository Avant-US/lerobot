MERGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$MERGE_ROOT"

python lerobot/bt/dm0/dm0_ws_server.py \
    --ckpt ${MERGE_ROOT}/lerobot/outputs/bt/dm0/train-20260511_034518_3092778_1356223099/checkpoints/010000/pretrained_model \
    --host 0.0.0.0 --port 8000