#!/usr/bin/env python3
"""
Vertex AI 分布式训练入口脚本。

在 Vertex AI CustomJob 中运行时，Vertex AI 会设置以下环境变量：
  - CLUSTER_SPEC: JSON 格式的集群拓扑信息
  - TF_CONFIG: TensorFlow 风格的集群配置（PyTorch 也可用于解析）
  - MASTER_ADDR / MASTER_PORT: torchrun 需要的主节点信息

本脚本：
  1. 解析 Vertex AI 提供的环境变量，设置 PyTorch 分布式所需的 env vars
  2. 通过 accelerate launch 或 torchrun 启动 lerobot-train
  3. 训练完成后将 checkpoint 上传到 GCS

用法（Vertex AI 容器内自动调用）：
  python train_vertex.py [--所有参数见下方]

也可本地测试：
  python train_vertex.py --local-test --steps 2 --batch-size 1
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
LOGGER = logging.getLogger(__name__)


def parse_vertex_cluster_spec() -> dict:
    """解析 Vertex AI 的 CLUSTER_SPEC 环境变量，返回分布式训练参数。"""
    cluster_spec_str = os.environ.get("CLUSTER_SPEC", "")
    tf_config_str = os.environ.get("TF_CONFIG", "")

    result = {
        "is_distributed": False,
        "world_size": 1,
        "node_rank": 0,
        "master_addr": "localhost",
        "master_port": "29500",
        "num_nodes": 1,
    }

    # 优先用 CLUSTER_SPEC
    if cluster_spec_str:
        try:
            spec = json.loads(cluster_spec_str)
            LOGGER.info("CLUSTER_SPEC: %s", json.dumps(spec, indent=2))
            cluster = spec.get("cluster", {})
            task = spec.get("task", {})

            # 获取所有 worker 列表
            workers = cluster.get("workerpool0", [])
            if not workers:
                # 有时 key 是 "worker"
                workers = cluster.get("worker", [])

            if len(workers) > 1:
                result["is_distributed"] = True
                result["num_nodes"] = len(workers)
                result["world_size"] = len(workers)
                result["node_rank"] = task.get("index", 0)
                # 主节点是第一个 worker
                master = workers[0]
                if ":" in master:
                    result["master_addr"] = master.split(":")[0]
                    result["master_port"] = master.split(":")[1]
                else:
                    result["master_addr"] = master
        except json.JSONDecodeError:
            LOGGER.warning("无法解析 CLUSTER_SPEC: %s", cluster_spec_str)

    # 如果没有 CLUSTER_SPEC，尝试 TF_CONFIG
    elif tf_config_str:
        try:
            tf_config = json.loads(tf_config_str)
            LOGGER.info("TF_CONFIG: %s", json.dumps(tf_config, indent=2))
            cluster = tf_config.get("cluster", {})
            task = tf_config.get("task", {})

            workers = cluster.get("worker", [])
            chief = cluster.get("chief", [])
            all_nodes = chief + workers

            if len(all_nodes) > 1:
                result["is_distributed"] = True
                result["num_nodes"] = len(all_nodes)
                result["world_size"] = len(all_nodes)
                task_type = task.get("type", "worker")
                task_index = task.get("index", 0)
                if task_type == "chief":
                    result["node_rank"] = 0
                else:
                    result["node_rank"] = task_index + len(chief)

                master = all_nodes[0]
                if ":" in master:
                    result["master_addr"] = master.split(":")[0]
                    result["master_port"] = master.split(":")[1]
                else:
                    result["master_addr"] = master
        except json.JSONDecodeError:
            LOGGER.warning("无法解析 TF_CONFIG: %s", tf_config_str)

    # 环境变量覆盖
    if os.environ.get("MASTER_ADDR"):
        result["master_addr"] = os.environ["MASTER_ADDR"]
    if os.environ.get("MASTER_PORT"):
        result["master_port"] = os.environ["MASTER_PORT"]

    return result


def get_gpu_count() -> int:
    """获取当前节点的 GPU 数量。"""
    try:
        import torch
        count = torch.cuda.device_count()
        return max(count, 1)
    except Exception:
        return 1


def upload_to_gcs(local_dir: str, gcs_path: str) -> None:
    """将本地目录上传到 GCS。"""
    if not gcs_path.startswith("gs://"):
        LOGGER.info("GCS 路径未指定或无效，跳过上传: %s", gcs_path)
        return
    LOGGER.info("上传 checkpoint 到 GCS: %s -> %s", local_dir, gcs_path)
    try:
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", local_dir, gcs_path],
            check=True,
        )
        LOGGER.info("上传完成: %s", gcs_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        LOGGER.error("GCS 上传失败: %s", e)


def build_lerobot_train_args(args) -> list[str]:
    """构建 lerobot-train 的命令行参数。"""
    train_args = [
        f"--policy.type=str_groot",
        f"--policy.push_to_hub=false",
        f"--policy.device=cuda",
        f"--policy.freeze_vlm={str(args.freeze_vlm).lower()}",
        f"--policy.tune_vlm={str(not args.freeze_vlm).lower()}",
        f"--policy.tune_action_head=true",
        f"--dataset.repo_id={args.dataset_repo}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--eval_freq=0",
        f"--log_freq={args.log_freq}",
        f"--save_checkpoint={str(args.save_checkpoint).lower()}",
        f"--save_freq={args.save_freq}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
    ]

    if args.starvla_checkpoint:
        train_args.append(f"--policy.starvla_checkpoint={args.starvla_checkpoint}")

    if args.state_indices:
        indices_str = "[" + ",".join(str(i) for i in args.state_indices) + "]"
        train_args.append(f"--policy.state_indices={indices_str}")

    if args.attn_implementation:
        train_args.append(f"--policy.attn_implementation={args.attn_implementation}")

    if args.wandb:
        train_args.extend([
            f"--wandb.enable=true",
            f"--wandb.project={args.wandb_project}",
            f"--wandb.disable_artifact=true",
        ])
        if args.wandb_entity:
            train_args.append(f"--wandb.entity={args.wandb_entity}")

    if args.resume:
        train_args.append("--resume=true")
        if args.config_path:
            train_args.append(f"--config_path={args.config_path}")

    if args.lr:
        train_args.append(f"--policy.optimizer_lr={args.lr}")

    return train_args


def main():
    import argparse

    p = argparse.ArgumentParser(description="Vertex AI 分布式训练入口")

    # --- 训练参数 ---
    p.add_argument("--dataset-repo", default="HuggingFaceVLA/libero")
    p.add_argument("--starvla-checkpoint", default="StarVLA/Qwen3VL-GR00T-Bridge-RT-1")
    p.add_argument("--state-indices", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 7])
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--freeze-vlm", action="store_true", default=True)
    p.add_argument("--no-freeze-vlm", dest="freeze_vlm", action="store_false")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-freq", type=int, default=100)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--save-checkpoint", action="store_true", default=True)
    p.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false")
    p.add_argument("--job-name", default="str_groot_vertex")
    p.add_argument("--output-dir", default="/gcs/output/str_groot_vertex")

    # --- Resume ---
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--config-path", default=None)

    # --- WandB ---
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default="str_groot_vertex")
    p.add_argument("--wandb-entity", default=None)

    # --- GCS ---
    p.add_argument("--gcs-output", default="", help="GCS 路径，训练结束后上传 checkpoint")

    # --- 测试 ---
    p.add_argument("--local-test", action="store_true", help="本地单卡测试模式")

    args = p.parse_args()

    # 1) 解析集群信息
    cluster = parse_vertex_cluster_spec()
    LOGGER.info("集群信息: %s", json.dumps(cluster, indent=2))

    nproc_per_node = get_gpu_count()
    LOGGER.info("当前节点 GPU 数量: %d", nproc_per_node)

    # 2) 设置分布式环境变量
    os.environ["MASTER_ADDR"] = cluster["master_addr"]
    os.environ["MASTER_PORT"] = cluster["master_port"]
    os.environ.setdefault("NCCL_DEBUG", "INFO")

    # 3) 构建 lerobot-train 参数
    lerobot_args = build_lerobot_train_args(args)

    # 4) 决定启动方式
    if args.local_test:
        # 本地测试：直接调用 lerobot-train
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train"] + lerobot_args
    elif cluster["is_distributed"] or nproc_per_node > 1:
        # 分布式训练：使用 accelerate launch
        cmd = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--multi_gpu",
            f"--num_processes={nproc_per_node * cluster['num_nodes']}",
            f"--num_machines={cluster['num_nodes']}",
            f"--machine_rank={cluster['node_rank']}",
            f"--main_process_ip={cluster['master_addr']}",
            f"--main_process_port={cluster['master_port']}",
            "-m", "lerobot.scripts.lerobot_train",
        ] + lerobot_args
    else:
        # 单卡
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train"] + lerobot_args

    LOGGER.info("启动命令: %s", " ".join(cmd))

    # 5) 执行训练
    result = subprocess.run(cmd)

    # 6) 上传到 GCS（仅主节点）
    if cluster["node_rank"] == 0 and args.gcs_output:
        upload_to_gcs(args.output_dir, args.gcs_output)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
