#!/bin/bash
# PI0.5 Pre-Training on HuggingFaceVLA/community_dataset_v3
#
# Usage:
#   ./bt/ptpi05/run.sh convert          # Download & convert 37 episodes to v3.0
#   ./bt/ptpi05/run.sh convert-small    # Download & convert 5 episodes (quick test)
#   ./bt/ptpi05/run.sh debug            # Train from random init, 20 steps on converted data
#   ./bt/ptpi05/run.sh debug-dummy      # Generate dummy dataset, then train
#   ./bt/ptpi05/run.sh train            # Full pre-train from pretrained base
#   ./bt/ptpi05/run.sh help

set -e
cd "$(dirname "$0")/../.."

MODE=${1:-debug}
COMMUNITY_V30_DIR="data/community_pt_v30"
DUMMY_DIR="data/dummy_pt_pi05"

case "$MODE" in
    convert)
        echo "=== Convert: download 37 episodes from community_dataset_v3 -> v3.0 ==="
        python -m bt.ptpi05.convert_community \
            --output-dir "$COMMUNITY_V30_DIR" \
            --num-episodes 37
        ;;
    convert-small)
        echo "=== Convert-Small: download 5 episodes from community_dataset_v3 -> v3.0 ==="
        python -m bt.ptpi05.convert_community \
            --output-dir "$COMMUNITY_V30_DIR" \
            --num-episodes 5 \
            --max-probe 30
        ;;
    debug)
        echo "=== Debug: random-init PI05 pre-train on community data (v3.0) ==="
        if [ ! -d "$COMMUNITY_V30_DIR/meta" ]; then
            echo "--- Dataset not found. Converting 5 episodes first ---"
            python -m bt.ptpi05.convert_community \
                --output-dir "$COMMUNITY_V30_DIR" \
                --num-episodes 5 \
                --max-probe 30
        fi
        python -m bt.ptpi05.train_pi05 \
            --repo-id community_pt_v30 \
            --local-root "$COMMUNITY_V30_DIR" \
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
            --output-dir outputs/pt_pi05_debug
        ;;
    debug-dummy)
        echo "=== Debug-Dummy: generate local dummy dataset, then pre-train ==="
        python -m bt.sftpi05.gen_dummy_dataset \
            --output-dir "$DUMMY_DIR" \
            --num-episodes 5 \
            --frames-per-ep 80

        python -m bt.ptpi05.train_pi05 \
            --repo-id dummy_pt_pi05 \
            --local-root "$DUMMY_DIR" \
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
            --output-dir outputs/pt_pi05_debug_dummy
        ;;
    train)
        echo "=== Pre-Train PI05 on community data (v3.0) ==="
        if [ ! -d "$COMMUNITY_V30_DIR/meta" ]; then
            echo "--- Dataset not found. Converting 37 episodes first ---"
            python -m bt.ptpi05.convert_community \
                --output-dir "$COMMUNITY_V30_DIR" \
                --num-episodes 37
        fi
        python -m bt.ptpi05.train_pi05 \
            --repo-id community_pt_v30 \
            --local-root "$COMMUNITY_V30_DIR" \
            --no-pretrained \
            --dtype bfloat16 \
            --batch-size 4 \
            --steps 3000 \
            --lr 2.5e-5 \
            --warmup-steps 100 \
            --log-freq 50 \
            --save-freq 500 \
            --num-workers 4 \
            --gradient-checkpointing
        ;;
    help|*)
        echo "Usage: $0 {convert|convert-small|debug|debug-dummy|train}"
        echo "  convert       - Download 37 episodes from community_dataset_v3, convert to v3.0"
        echo "  convert-small - Download 5 episodes (quick test)"
        echo "  debug         - Quick test with converted data, random weights, 20 steps"
        echo "  debug-dummy   - Generate dummy dataset, then quick train"
        echo "  train         - Full pre-train from scratch"
        ;;
esac
