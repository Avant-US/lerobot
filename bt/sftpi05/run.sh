#!/bin/bash
# PI05 Fine-Tuning on Libero dataset (LeRobot v3.0 format)
#
# Usage:
#   ./bt/sftpi05/run.sh debug        # Fast debug: 2 episodes, random weights, 20 steps
#   ./bt/sftpi05/run.sh debug-dummy   # Even faster: generate + use local dummy dataset
#   ./bt/sftpi05/run.sh train         # Real fine-tune from pretrained
#   ./bt/sftpi05/run.sh convert [N]   # Convert N libero episodes to v3.0 (default: all)
#   ./bt/sftpi05/run.sh help

set -e
cd "$(dirname "$0")/../.."

MODE=${1:-debug}
LIBERO_V30_DIR="data/libero_v30"

case "$MODE" in
    convert)
        N_EPISODES="${2:-}"
        echo "=== Convert libero v2.0 -> v3.0 ==="
        CONVERT_ARGS="--output-dir $LIBERO_V30_DIR"
        if [ -n "$N_EPISODES" ]; then
            CONVERT_ARGS="$CONVERT_ARGS --max-episodes $N_EPISODES"
        fi
        python -m bt.sftpi05.convert_libero $CONVERT_ARGS
        ;;
    debug)
        echo "=== Debug: random-init PI05 on 2 libero episodes (v3.0) ==="
        # Step 1: convert 2 episodes if not already done
        if [ ! -d "$LIBERO_V30_DIR/meta" ]; then
            echo "--- Converting 2 libero episodes to v3.0 first ---"
            python -m bt.sftpi05.convert_libero \
                --output-dir "$LIBERO_V30_DIR" \
                --max-episodes 2
        fi
        # Step 2: train
        python -m bt.sftpi05.train_pi05 \
            --repo-id physical-intelligence/libero \
            --local-root "$LIBERO_V30_DIR" \
            --max-episodes 2 \
            --no-pretrained \
            --dtype float32 \
            --batch-size 2 \
            --steps 20 \
            --lr 1e-4 \
            --warmup-steps 2 \
            --log-freq 5 \
            --save-freq 20 \
            --num-workers 0 \
            --gradient-checkpointing \
            --output-dir outputs/sft_pi05_debug
        ;;
    debug-dummy)
        echo "=== Debug-Dummy: generate local dummy dataset, then train ==="
        DUMMY_DIR="data/dummy_pi05_sft"
        python -m bt.sftpi05.gen_dummy_dataset \
            --output-dir "$DUMMY_DIR" \
            --num-episodes 3 \
            --frames-per-ep 60

        python -m bt.sftpi05.train_pi05 \
            --repo-id dummy_pi05_sft \
            --local-root "$DUMMY_DIR" \
            --max-episodes 3 \
            --no-pretrained \
            --dtype float32 \
            --batch-size 2 \
            --steps 20 \
            --lr 1e-4 \
            --warmup-steps 2 \
            --log-freq 5 \
            --save-freq 20 \
            --num-workers 0 \
            --gradient-checkpointing \
            --output-dir outputs/sft_pi05_debug_dummy
        ;;
    train)
        echo "=== Fine-tune PI05 on libero (pretrained base, v3.0) ==="
        if [ ! -d "$LIBERO_V30_DIR/meta" ]; then
            echo "--- Converting libero dataset to v3.0 first ---"
            python -m bt.sftpi05.convert_libero --output-dir "$LIBERO_V30_DIR"
        fi
        python -m bt.sftpi05.train_pi05 \
            --repo-id physical-intelligence/libero \
            --local-root "$LIBERO_V30_DIR" \
            --pretrained-path lerobot/pi05_base \
            --dtype bfloat16 \
            --batch-size 4 \
            --steps 3000 \
            --lr 2.5e-5 \
            --warmup-steps 100 \
            --log-freq 50 \
            --save-freq 500 \
            --num-workers 4 \
            --gradient-checkpointing \
            --freeze-vision-encoder
        ;;
    help|*)
        echo "Usage: $0 {debug|debug-dummy|train|convert}"
        echo "  debug       - Quick test with 2 libero episodes, random weights, 20 steps"
        echo "  debug-dummy - Generate a local dummy dataset, then quick train"
        echo "  train       - Full fine-tune from lerobot/pi05_base"
        echo "  convert [N] - Convert N libero episodes from v2.0 to v3.0"
        ;;
esac
