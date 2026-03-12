#!/bin/bash
# SmolVLA Pre-Training on HuggingFaceVLA/community_dataset_v3
#
# Usage:
#   ./bt/ptsmolvla/run.sh convert          # Download & convert 37 episodes to v3.0
#   ./bt/ptsmolvla/run.sh convert-small    # Download & convert 5 episodes (quick test)
#   ./bt/ptsmolvla/run.sh debug            # Train from random init, 20 steps on converted data
#   ./bt/ptsmolvla/run.sh train            # Full pre-train from scratch
#   ./bt/ptsmolvla/run.sh help

set -e
cd "$(dirname "$0")/../.."

MODE=${1:-debug}
COMMUNITY_V30_DIR="data/community_pt_smolvla"

case "$MODE" in
    convert)
        echo "=== Convert: download 37 episodes from community_dataset_v3 -> v3.0 ==="
        python -m bt.ptsmolvla.convert_community \
            --output-dir "$COMMUNITY_V30_DIR" \
            --num-episodes 37
        ;;
    convert-small)
        echo "=== Convert-Small: download 5 episodes from community_dataset_v3 -> v3.0 ==="
        python -m bt.ptsmolvla.convert_community \
            --output-dir "$COMMUNITY_V30_DIR" \
            --num-episodes 5 \
            --max-probe 30
        ;;
    debug)
        echo "=== Debug: random-init SmolVLA pre-train on community data (v3.0) ==="
        if [ ! -d "$COMMUNITY_V30_DIR/meta" ]; then
            echo "--- Dataset not found. Converting 5 episodes first ---"
            python -m bt.ptsmolvla.convert_community \
                --output-dir "$COMMUNITY_V30_DIR" \
                --num-episodes 5 \
                --max-probe 30
        fi
        python -m bt.ptsmolvla.train_smolvla \
            --repo-id community_pt_smolvla \
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
            --output-dir outputs/pt_smolvla_debug
        ;;
    train)
        echo "=== Pre-Train SmolVLA on community data (v3.0) ==="
        if [ ! -d "$COMMUNITY_V30_DIR/meta" ]; then
            echo "--- Dataset not found. Converting 37 episodes first ---"
            python -m bt.ptsmolvla.convert_community \
                --output-dir "$COMMUNITY_V30_DIR" \
                --num-episodes 37
        fi
        python -m bt.ptsmolvla.train_smolvla \
            --repo-id community_pt_smolvla \
            --local-root "$COMMUNITY_V30_DIR" \
            --no-pretrained \
            --dtype bfloat16 \
            --batch-size 4 \
            --steps 3000 \
            --lr 1e-4 \
            --warmup-steps 100 \
            --log-freq 50 \
            --save-freq 500 \
            --num-workers 4 \
            --gradient-checkpointing
        ;;
    help|*)
        echo "Usage: $0 {convert|convert-small|debug|train}"
        echo "  convert       - Download 37 episodes from community_dataset_v3, convert to v3.0"
        echo "  convert-small - Download 5 episodes (quick test)"
        echo "  debug         - Quick test with converted data, random weights, 20 steps"
        echo "  train         - Full pre-train from scratch"
        ;;
esac
