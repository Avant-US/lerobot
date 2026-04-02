#!/usr/bin/env python3
"""
提交 Vertex AI CustomJob 进行 StrGroot 分布式训练。

使用 Vertex AI Python SDK 创建和提交自定义训练作业。
支持单节点多卡和多节点多卡分布式训练。

用法:
  # 单节点 8xA100 训练
  python bt/vtx_1/submit_vertex_job.py

  # 2 节点 x 8xA100 分布式训练
  python bt/vtx_1/submit_vertex_job.py --num-nodes 2

  # 自定义参数
  python bt/vtx_1/submit_vertex_job.py \
    --num-nodes 4 \
    --gpu-type NVIDIA_A100_80GB \
    --gpu-count 8 \
    --steps 50000 \
    --batch-size 64

  # dry-run 模式（只打印配置，不提交）
  python bt/vtx_1/submit_vertex_job.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
LOGGER = logging.getLogger(__name__)


# Vertex AI 支持的 GPU 类型
GPU_TYPES = {
    "T4": "NVIDIA_TESLA_T4",
    "V100": "NVIDIA_TESLA_V100",
    "A100": "NVIDIA_TESLA_A100",
    "A100_80GB": "NVIDIA_A100_80GB",
    "L4": "NVIDIA_L4",
    "H100": "NVIDIA_H100_80GB",
    "H100_MEGA": "NVIDIA_H100_MEGA_80GB",
}

# GPU 类型对应的推荐机器类型
GPU_MACHINE_MAP = {
    "NVIDIA_TESLA_T4": "n1-standard-16",
    "NVIDIA_TESLA_V100": "n1-standard-16",
    "NVIDIA_TESLA_A100": "a2-highgpu-8g",
    "NVIDIA_A100_80GB": "a2-ultragpu-8g",
    "NVIDIA_L4": "g2-standard-96",
    "NVIDIA_H100_80GB": "a3-highgpu-8g",
    "NVIDIA_H100_MEGA_80GB": "a3-megagpu-8g",
}


def parse_args():
    p = argparse.ArgumentParser(description="提交 Vertex AI StrGroot 训练作业")

    # --- GCP 配置 ---
    p.add_argument("--project", default=None, help="GCP 项目 ID（默认使用 gcloud 当前项目）")
    p.add_argument("--region", default="us-central1", help="Vertex AI 区域")
    p.add_argument("--staging-bucket", default=None, help="GCS staging bucket (gs://...)")

    # --- 镜像 ---
    p.add_argument("--image-uri", default=None, help="完整镜像 URI（覆盖自动构建的 URI）")
    p.add_argument("--repo-name", default="lerobot-training")
    p.add_argument("--image-name", default="str-groot-vertex")
    p.add_argument("--image-tag", default="latest")

    # --- 计算资源 ---
    p.add_argument("--num-nodes", type=int, default=1, help="训练节点数")
    p.add_argument("--machine-type", default=None, help="机器类型（自动根据 GPU 推断）")
    p.add_argument("--gpu-type", default="A100_80GB", choices=list(GPU_TYPES.keys()),
                    help="GPU 类型简称")
    p.add_argument("--gpu-count", type=int, default=8, help="每节点 GPU 数")
    p.add_argument("--boot-disk-size-gb", type=int, default=200)

    # --- 训练参数（传递给容器）---
    p.add_argument("--dataset-repo", default="HuggingFaceVLA/libero")
    p.add_argument("--starvla-checkpoint", default="StarVLA/Qwen3VL-GR00T-Bridge-RT-1")
    p.add_argument("--state-indices", default="0,1,2,3,4,5,7")
    p.add_argument("--freeze-vlm", action="store_true", default=True)
    p.add_argument("--no-freeze-vlm", dest="freeze_vlm", action="store_false")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-freq", type=int, default=100)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--job-name-prefix", default="str-groot")

    # --- GCS 输出 ---
    p.add_argument("--gcs-output", default="", help="训练完成后上传 checkpoint 到此 GCS 路径")
    p.add_argument("--output-dir", default="/gcs/output/str_groot_vertex",
                    help="容器内输出目录")

    # --- WandB ---
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default="str_groot_vertex")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-api-key", default=None, help="WandB API key（也可通过 WANDB_API_KEY 环境变量设置）")

    # --- 其他 ---
    p.add_argument("--service-account", default=None, help="用于训练作业的 Service Account")
    p.add_argument("--network", default=None, help="VPC 网络（如需要 peering）")
    p.add_argument("--tensorboard", default=None, help="Vertex AI Tensorboard 实例名")
    p.add_argument("--dry-run", action="store_true", help="仅打印配置，不提交作业")

    return p.parse_args()


def get_default_project():
    """从 gcloud config 获取默认项目。"""
    import subprocess
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def build_container_args(args) -> list[str]:
    """构建传递给容器 ENTRYPOINT 的参数列表。"""
    container_args = [
        f"--dataset-repo={args.dataset_repo}",
        f"--starvla-checkpoint={args.starvla_checkpoint}",
        f"--steps={args.steps}",
        f"--batch-size={args.batch_size}",
        f"--num-workers={args.num_workers}",
        f"--lr={args.lr}",
        f"--log-freq={args.log_freq}",
        f"--save-freq={args.save_freq}",
        f"--output-dir={args.output_dir}",
        f"--job-name={args.job_name_prefix}",
    ]

    if args.state_indices:
        indices = [int(x) for x in args.state_indices.split(",")]
        container_args.append("--state-indices")
        container_args.extend([str(i) for i in indices])

    if args.freeze_vlm:
        container_args.append("--freeze-vlm")

    if args.gcs_output:
        container_args.append(f"--gcs-output={args.gcs_output}")

    if args.wandb:
        container_args.extend([
            "--wandb",
            f"--wandb-project={args.wandb_project}",
        ])
        if args.wandb_entity:
            container_args.append(f"--wandb-entity={args.wandb_entity}")

    return container_args


def main():
    args = parse_args()

    # 确定项目 ID
    project = args.project or get_default_project()
    if not project:
        LOGGER.error("请指定 --project 或设置 gcloud 默认项目")
        return

    # 确定镜像 URI
    if args.image_uri:
        image_uri = args.image_uri
    else:
        image_uri = (
            f"{args.region}-docker.pkg.dev/{project}/"
            f"{args.repo_name}/{args.image_name}:{args.image_tag}"
        )

    # GPU 类型
    gpu_type_full = GPU_TYPES[args.gpu_type]
    machine_type = args.machine_type or GPU_MACHINE_MAP.get(gpu_type_full, "n1-standard-16")

    # 作业名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_display_name = f"{args.job_name_prefix}-{args.num_nodes}x{args.gpu_count}gpu-{timestamp}"

    # 容器参数
    container_args = build_container_args(args)

    # 环境变量
    env_vars = {
        "NCCL_DEBUG": "INFO",
        "CUDA_LAUNCH_BLOCKING": "0",
    }
    if args.wandb_api_key:
        env_vars["WANDB_API_KEY"] = args.wandb_api_key

    LOGGER.info("=" * 60)
    LOGGER.info("  Vertex AI StrGroot 训练作业配置")
    LOGGER.info("=" * 60)
    LOGGER.info("项目:       %s", project)
    LOGGER.info("区域:       %s", args.region)
    LOGGER.info("镜像:       %s", image_uri)
    LOGGER.info("作业名:     %s", job_display_name)
    LOGGER.info("节点数:     %d", args.num_nodes)
    LOGGER.info("机器类型:   %s", machine_type)
    LOGGER.info("GPU:        %d x %s / node", args.gpu_count, gpu_type_full)
    LOGGER.info("总 GPU 数:  %d", args.num_nodes * args.gpu_count)
    LOGGER.info("训练步数:   %d", args.steps)
    LOGGER.info("Batch size: %d (per node)", args.batch_size)
    LOGGER.info("容器参数:   %s", " ".join(container_args))
    LOGGER.info("=" * 60)

    if args.dry_run:
        LOGGER.info("DRY RUN 模式 — 不提交作业。")
        return

    # --- 提交作业 ---
    try:
        from google.cloud import aiplatform
    except ImportError:
        LOGGER.error("请安装 google-cloud-aiplatform: pip install google-cloud-aiplatform>=1.60.0")
        return

    aiplatform.init(
        project=project,
        location=args.region,
        staging_bucket=args.staging_bucket,
    )

    # 构建 worker pool specs
    worker_pool_specs = []

    # Primary worker (workerpool0)
    primary_spec = {
        "machine_spec": {
            "machine_type": machine_type,
            "accelerator_type": gpu_type_full,
            "accelerator_count": args.gpu_count,
        },
        "replica_count": 1,
        "disk_spec": {
            "boot_disk_type": "pd-ssd",
            "boot_disk_size_gb": args.boot_disk_size_gb,
        },
        "container_spec": {
            "image_uri": image_uri,
            "args": container_args,
            "env": [{"name": k, "value": v} for k, v in env_vars.items()],
        },
    }
    worker_pool_specs.append(primary_spec)

    # Additional workers (workerpool1) for multi-node
    if args.num_nodes > 1:
        worker_spec = {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": gpu_type_full,
                "accelerator_count": args.gpu_count,
            },
            "replica_count": args.num_nodes - 1,
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": args.boot_disk_size_gb,
            },
            "container_spec": {
                "image_uri": image_uri,
                "args": container_args,
                "env": [{"name": k, "value": v} for k, v in env_vars.items()],
            },
        }
        worker_pool_specs.append(worker_spec)

    # 创建 CustomJob
    custom_job = aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=worker_pool_specs,
    )

    LOGGER.info("提交训练作业: %s", job_display_name)

    custom_job.run(
        service_account=args.service_account,
        network=args.network,
        tensorboard=args.tensorboard,
        sync=False,  # 异步提交，不阻塞
    )

    LOGGER.info("作业已提交!")
    LOGGER.info("作业名: %s", custom_job.display_name)
    LOGGER.info("资源名: %s", custom_job.resource_name)
    LOGGER.info("")
    LOGGER.info("查看作业状态:")
    LOGGER.info("  gcloud ai custom-jobs describe %s --region=%s --project=%s",
                custom_job.resource_name.split("/")[-1], args.region, project)
    LOGGER.info("")
    LOGGER.info("查看日志:")
    LOGGER.info("  gcloud ai custom-jobs stream-logs %s --region=%s --project=%s",
                custom_job.resource_name.split("/")[-1], args.region, project)


if __name__ == "__main__":
    main()
