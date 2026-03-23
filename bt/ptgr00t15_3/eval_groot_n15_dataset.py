#!/usr/bin/env python3
"""
参考 lerobot_eval.py 与 str_groot 的训练/评估脚本，使用 groot2 policy 在 LIBERO 上评估 GR00T N1.5。

输出：
- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-Long
- 四项平均分

python bt/ptgr00t15_3/eval_groot_n15_dataset.py --n-episodes 1 --batch-size 1 --max-tasks-per-suite 1 --max-parallel-tasks 1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
LOGGER = logging.getLogger(__name__)

ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_DISPLAY = {
    "libero_spatial": "LIBERO-Spatial",
    "libero_object": "LIBERO-Object",
    "libero_goal": "LIBERO-Goal",
    "libero_10": "LIBERO-Long",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 groot2 评估 GR00T N1.5 在 LIBERO 的分数")
    parser.add_argument(
        "--policy-path",
        default="nvidia/GR00T-N1.5-3B",
        help="GR00T N1.5 模型路径（HF repo id 或本地 pretrained_model 目录）",
    )
    parser.add_argument("--dataset-repo", default="HuggingFaceVLA/libero", help="用于加载归一化统计量的dataset")
    parser.add_argument("--dataset-root", default=None, help="dataset 本地缓存路径，可不填")
    parser.add_argument("--suites", nargs="*", default=None, help="默认评估四个 suite")
    parser.add_argument("--n-episodes", type=int, default=10, help="每个 task 的评估 episode 数")
    parser.add_argument("--batch-size", type=int, default=1, help="每个 task 并行环境数")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default=None, help="cuda / cpu / cuda:0")
    parser.add_argument("--control-mode", default="relative", choices=["relative", "absolute"])
    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--no-use-bf16", dest="use_bf16", action="store_false")
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument(
        "--max-tasks-per-suite",
        type=int,
        default=None,
        help="调试加速选项：每个 suite 只评估前 N 个 task（默认评估 suite 全部 task）",
    )
    parser.add_argument("--output-dir", default="outputs/bt/ptgr00t15_3/eval_libero")
    parser.add_argument("--dry-run", action="store_true", help="仅验证 env/model/processor 构建")
    return parser.parse_args()


def validate_suites(suites: list[str]) -> list[str]:
    bad = [s for s in suites if s not in ALL_SUITES]
    if bad:
        raise ValueError(f"未知 suites: {bad}，可选值: {ALL_SUITES}")
    return suites


def maybe_trim_env_tasks(envs: dict, max_tasks_per_suite: int | None) -> dict:
    if max_tasks_per_suite is None:
        return envs
    if max_tasks_per_suite <= 0:
        raise ValueError("--max-tasks-per-suite 必须 > 0")

    trimmed = {}
    for suite, suite_envs in envs.items():
        task_ids = sorted(suite_envs.keys())
        keep = set(task_ids[:max_tasks_per_suite])
        trimmed[suite] = {}
        for task_id, vec_env in suite_envs.items():
            if task_id in keep:
                trimmed[suite][task_id] = vec_env
            else:
                vec_env.close()
        LOGGER.info(
            "Suite %s: 保留 %d/%d 个 tasks",
            suite,
            len(trimmed[suite]),
            len(task_ids),
        )
    return trimmed


def print_scores(info: dict, suites: list[str]) -> None:
    per_group = info.get("per_group", {})
    scores = {}

    print("\n" + "=" * 62)
    print("  GR00T N1.5 (groot2) - LIBERO Evaluation")
    print("=" * 62)
    for suite in suites:
        name = SUITE_DISPLAY[suite]
        if suite in per_group:
            score = float(per_group[suite].get("pc_success", float("nan")))
            scores[name] = score
            print(f"  {name:<18s} {score:7.2f}%")
        else:
            print(f"  {name:<18s} {'N/A':>7s}")

    valid_scores = [v for v in scores.values() if not np.isnan(v)]
    avg = float(np.mean(valid_scores)) if valid_scores else float("nan")
    print("-" * 62)
    print(f"  {'Average':<18s} {avg:7.2f}%")
    print("=" * 62 + "\n")


def main() -> None:
    args = parse_args()
    suites = validate_suites(args.suites or ALL_SUITES)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.split(":")[0]

    # 延迟导入，加快 --help / 参数错误返回
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.groot2.configuration_groot import GrootConfig
    from lerobot.scripts.lerobot_eval import eval_policy_all

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Suites: %s", suites)
    LOGGER.info("Creating LIBERO envs (batch_size=%d)", args.batch_size)
    env_cfg = LiberoEnv(
        task=",".join(suites),
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        control_mode=args.control_mode,
    )
    envs = make_env(env_cfg, n_envs=args.batch_size)
    envs = maybe_trim_env_tasks(envs, args.max_tasks_per_suite)

    # 使用 groot2 policy（不修改原 groot）
    policy_cfg = GrootConfig(
        push_to_hub=False,
        repo_id=None,
        base_model_path=args.policy_path,
        device=device,
        use_bf16=args.use_bf16,
    )

    LOGGER.info("Loading groot2 policy from: %s", args.policy_path)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    LOGGER.info("Loading dataset stats from: %s", args.dataset_repo)
    ds_meta = LeRobotDatasetMetadata(repo_id=args.dataset_repo, root=args.dataset_root)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        dataset_stats=ds_meta.stats,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    if args.dry_run:
        LOGGER.info("dry-run 完成：env + policy + processor 已构建成功。")
        close_envs(envs)
        return

    LOGGER.info(
        "Start eval: n_episodes=%d, seed=%d, max_parallel_tasks=%d",
        args.n_episodes,
        args.seed,
        args.max_parallel_tasks,
    )
    t0 = time.time()
    amp_ctx = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if args.use_bf16 and device_type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            start_seed=args.seed,
            max_parallel_tasks=args.max_parallel_tasks,
        )
    close_envs(envs)
    LOGGER.info("Eval finished in %.1fs", time.time() - t0)

    print_scores(info, suites)

    result_file = output_dir / "eval_results.json"
    result_file.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Saved full eval results: %s", result_file)


if __name__ == "__main__":
    main()
