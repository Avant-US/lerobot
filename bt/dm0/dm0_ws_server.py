"""DM0 over OpenPI-compatible WebSocket protocol.

兼容 r1pro_chassis 的 WebSocketClientEngine + OpenPIProcessor 客户端。
客户端发的 obs（msgpack + numpy）：
    {head_rgb: HWC uint8/float, left_wrist_rgb, right_wrist_rgb,
     state: (23,) float32, prompt: str}
服务端回的 action：
    {actions: (T, >=23) float32,
     policy_timing: {infer_ms: float},
     server_timing: {infer_ms: float}}     # policy_server 自动补
"""
from __future__ import annotations

import argparse
import asyncio
import http
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import websockets.asyncio.server as _server
import websockets.frames

# 让脚本在 dm0 仓里独立跑
_LEROBOT_SRC = Path(__file__).resolve().parents[2] / "src"  # 视实际放置位置而定
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

from lerobot.policies.dm0.modeling_dm0 import DM0Policy
from lerobot.utils.constants import OBS_STATE

# 直接用 r1pro_chassis 那份 hook（也可以把 23 行 msgpack.py 拷过来避免依赖）
_DM0_DIR = Path(__file__).resolve().parent
if str(_DM0_DIR) not in sys.path:
    sys.path.insert(0, str(_DM0_DIR))
from utils.msgpack import Packer, unpackb

logger = logging.getLogger(__name__)


# ─── DM0 适配层 ──────────────────────────────────────────────

# 训练数据里相机的 key 名字（dexbotic 习惯命名，跟 dm0 ckpt 内部对齐）
HEAD_KEY  = "observation.images.head_rgb"
LEFT_KEY  = "observation.images.left_wrist_rgb"
RIGHT_KEY = "observation.images.right_wrist_rgb"


def _img_hwc_to_chw_torch(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """OpenPI/客户端发的图是 HWC uint8 或 float[0,1]。DM0 要 [B, C, H, W]，[0,1] float。"""
    t = torch.from_numpy(arr).to(device)
    if t.ndim == 2:
        t = t.unsqueeze(-1).expand(-1, -1, 3)
    if t.ndim == 3 and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)
    t = t.float()
    if t.max() > 1.5:
        t = t / 255.0
    return t.unsqueeze(0).contiguous()  # [1, C, H, W]


def _build_dm0_batch(obs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    state = torch.as_tensor(obs["state"], dtype=torch.float32, device=device).reshape(1, -1)
    prompt = obs.get("prompt", "")
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8", errors="replace")

    return {
        OBS_STATE: state,                                              # [1, 23]
        HEAD_KEY:  _img_hwc_to_chw_torch(obs["head_rgb"], device),
        LEFT_KEY:  _img_hwc_to_chw_torch(obs["left_wrist_rgb"], device),
        RIGHT_KEY: _img_hwc_to_chw_torch(obs["right_wrist_rgb"], device),
        "task": [str(prompt) if prompt else ""],
    }


class DM0Adapter:
    def __init__(
        self,
        ckpt: str | Path,
        *,
        device: str | torch.device = "cuda",
        peft_base: str | None = None,
        merge_lora: bool = False,
    ) -> None:
        self.device = torch.device(device)
        ckpt = Path(ckpt).expanduser().resolve()
        if not ckpt.is_dir():
            raise FileNotFoundError(f"DM0 ckpt dir missing: {ckpt}")

        print(f"[dm0_ws] loading weights from {ckpt} ...", flush=True)
        if (ckpt / "adapter_config.json").is_file():
            from peft import PeftConfig, PeftModel
            peft_cfg = PeftConfig.from_pretrained(str(ckpt))
            base_dir = peft_base or peft_cfg.base_model_name_or_path
            base = DM0Policy.from_pretrained(str(base_dir), strict=False)
            policy: Any = PeftModel.from_pretrained(base, str(ckpt), config=peft_cfg)
            if merge_lora:
                policy = policy.merge_and_unload()
        else:
            policy = DM0Policy.from_pretrained(str(ckpt), strict=False)

        policy.eval().to(self.device)
        self._policy = policy
        self._dm0 = self._unwrap(policy)
        logger.info(f"[dm0_ws] loaded from {ckpt} on {self.device}")
        print(f"[dm0_ws] policy on GPU/CPU done: {self.device}", flush=True)

    @staticmethod
    def _unwrap(p: Any) -> DM0Policy:
        for _ in range(4):
            if isinstance(p, DM0Policy):
                return p
            p = getattr(p, "base_model", p)
            p = getattr(p, "model", p)
        if isinstance(p, DM0Policy):
            return p
        raise TypeError(f"Cannot unwrap DM0Policy from {type(p).__name__}")

    @torch.inference_mode()
    def predict(self, obs: dict[str, Any]) -> dict[str, Any]:
        batch = _build_dm0_batch(obs, self.device)
        t0 = time.monotonic()
        actions = self._dm0.predict_action_chunk(batch)  # [1, T, A]
        dt_ms = (time.monotonic() - t0) * 1000.0
        # 客户端会取 batch["actions"]，并按 23 维拆 left_arm/right_arm/...
        return {
            "actions": actions.squeeze(0).float().cpu().numpy(),  # [T, A]
            "policy_timing": {"infer_ms": dt_ms},
        }


# ─── WebSocket Server（结构完全照抄 serving/policy_server.py） ────

class Dm0PolicyServer:
    def __init__(self, adapter: DM0Adapter, host: str, port: int, metadata: dict | None = None):
        self._adapter = adapter
        self._host = host
        self._port = port
        self._metadata = metadata or {"policy": "dm0", "version": 1}

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with _server.serve(
            self._handler, self._host, self._port,
            compression=None, max_size=None,
            process_request=_health_check,
        ) as server:
            health_host = "127.0.0.1" if self._host in ("0.0.0.0", "::", "[::]") else self._host
            logger.info(f"[dm0_ws] listening on ws://{self._host}:{self._port}")
            print(
                f"[dm0_ws] listening ws://{self._host}:{self._port}  "
                f"healthz: curl -s http://{health_host}:{self._port}/healthz",
                flush=True,
            )
            await server.serve_forever()

    async def _handler(self, ws: _server.ServerConnection) -> None:
        logger.info(f"[dm0_ws] client connected: {ws.remote_address}")
        packer = Packer()
        await ws.send(packer.pack(self._metadata))

        prev_total_time: float | None = None
        while True:
            try:
                t_start = time.monotonic()
                obs = unpackb(await ws.recv())

                t_infer = time.monotonic()
                action = await asyncio.to_thread(self._adapter.predict, obs)
                infer_ms = (time.monotonic() - t_infer) * 1000.0

                action.setdefault("server_timing", {})["infer_ms"] = infer_ms
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await ws.send(packer.pack(action))
                prev_total_time = time.monotonic() - t_start

            except __import__("websockets").ConnectionClosed:
                logger.info(f"[dm0_ws] client disconnected: {ws.remote_address}")
                break
            except Exception:
                tb = traceback.format_exc()
                logger.error(tb)
                await ws.send(tb)
                await ws.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise


def _health_check(conn: _server.ServerConnection, req: _server.Request):
    if req.path == "/healthz":
        return conn.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ─── 入口 ────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="DM0 pretrained_model dir")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--peft-base", default=None)
    ap.add_argument("--merge-lora", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(
        f"[dm0_ws] start ckpt={args.ckpt} ws://{args.host}:{args.port} device={args.device}",
        flush=True,
    )
    adapter = DM0Adapter(args.ckpt, device=args.device,
                         peft_base=args.peft_base, merge_lora=args.merge_lora)
    print("[dm0_ws] starting websocket loop ...", flush=True)
    Dm0PolicyServer(adapter, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()