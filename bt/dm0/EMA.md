# DM0 EMA 机制说明

本文说明 `train_dm0_r1_pro.py` 中 EMA（Exponential Moving Average，指数滑动平均）是如何接入 DM0 训练、推理和 checkpoint 流程的。

## 1. 如何开启 EMA

训练入口在 `lerobot/bt/dm0/train_dm0_r1_pro.py` 中暴露了参数：

```bash
--ema-decay
```

默认值为 `None`，表示不启用 EMA：

```python
p.add_argument("--ema-decay", type=float, default=None, help="EMA decay(0<decay<1). None = disabled.")
```

启用示例：

```bash
accelerate launch bt/dm0/train_dm0_r1_pro.py \
  --task=train \
  --dataset-repo=local/r1_pro_chassis_v3 \
  --dataset-root=/path/to/dataset \
  --norm-stats-path=./norm_stats/r1_pro_chassis_v3.json \
  --dm0-base=./checkpoints/DM0-base \
  --ema-decay=0.999
```

`ema_decay` 必须在 `0 < decay < 1` 范围内。常见取值是 `0.999` 或 `0.9999`。数值越接近 1，EMA 参数变化越慢，平滑效果越强。

## 2. EMA 的基本原理

EMA 的核心思想是：训练时模型参数每一步都会被 optimizer 更新，但这些“即时参数”可能有较大的抖动；EMA 额外维护一份“平滑参数”，它不是通过反向传播直接训练出来的，而是把历史参数按指数衰减做平均。

对于某个可训练参数 `w`，训练过程中同时存在两份值：

```text
w_train: optimizer 正在更新的真实训练参数
w_ema:   由 EMA 维护的平滑参数
```

每个 optimizer step 后，用当前的 `w_train` 更新 `w_ema`：

```text
w_ema(t) = decay * w_ema(t-1) + (1 - decay) * w_train(t)
```

例如 `decay=0.999` 时，本 step 当前参数只占 `0.001`，历史 EMA 占 `0.999`。所以 EMA 参数不会剧烈跟随单个 batch 的梯度变化，而是更像最近很多 step 参数状态的加权平均。

这个“指数”体现在历史权重会按时间不断衰减。展开公式可以看到：

```text
w_ema(t)
  = (1 - decay) * w_train(t)
  + decay * (1 - decay) * w_train(t-1)
  + decay^2 * (1 - decay) * w_train(t-2)
  + ...
```

越新的参数权重越大，越旧的参数权重越小，但旧参数不会突然消失，而是指数级衰减。

### EMA 与普通平均的区别

普通平均会把历史上所有 step 等权平均，早期很差的参数也会长期影响结果。EMA 则更适合训练过程，因为它更重视最近的参数，同时又能过滤掉单个 batch 带来的尖峰波动。

可以把 EMA 理解成“低通滤波器”：optimizer 负责快速追踪梯度方向，EMA 负责把这条参数轨迹变平滑。最终用于推理的 EMA 权重通常比最后一个 step 的瞬时权重更稳定。

### EMA 在深度学习里的作用

EMA 不改变 loss 计算，也不参与反向传播。它的作用主要发生在训练参数更新之后，以及推理 / 保存模型之前：

- 训练时：optimizer 正常更新 `w_train`，EMA 只是旁路记录 `w_ema`。
- 反向传播时：梯度仍然只作用在 `w_train` 上，不对 `w_ema` 求梯度。
- 推理时：临时使用 `w_ema`，通常能得到更平滑、更稳定的行为。
- 保存时：把 `w_ema` 保存成推理权重，同时额外保存 `w_train` 以便继续训练。

对机器人策略模型来说，EMA 的意义通常比较直接：action 输出对权重抖动很敏感，最后一个训练 step 的参数可能刚好被某个 batch 拉偏；EMA 相当于用最近一段训练轨迹的平均策略做推理，动作序列更不容易出现高频抖动。

## 3. 配置传递链路

入口脚本只负责接收 `--ema-decay`，然后把它写入 `DM0Config`：

```python
DM0Config(
    ...
    ema_decay=args.ema_decay,
    ...
)
```

三个训练模式都会传递该字段：

- `--task=train`：phase-1，冻结 ViT，训练 LLM / projector / action head。
- `--task=lora_train`：phase-2，冻结主体，在 ViT 上挂 LoRA。
- `--task=full_train`：全量微调。

`DM0Config` 中对应字段定义在 `lerobot/src/lerobot/policies/dm0/configuration_dm0.py`：

```python
ema_decay: float | None = None  # EMA decay. None = disabled.
```

所以 EMA 是否启用完全由 policy config 控制。

## 4. EMA 状态在哪里初始化

EMA 状态维护在 `DM0Policy` 内部，相关文件是：

```text
lerobot/src/lerobot/policies/dm0/modeling_dm0.py
```

初始化时只创建占位状态：

```python
self._ema_params: dict[str, torch.Tensor] | None = None
self._ema_active: bool = False
```

真正的 EMA 参数不是在 `__init__` 里立即创建，而是在第一次 `update()` 时延迟初始化：

```python
def _init_ema(self):
    self._ema_params = {}
    for name, param in self.named_parameters():
        if param.requires_grad:
            self._ema_params[name] = param.data.clone()
```

这样做的目的：

- 等 PEFT / LoRA 包装完成后，再记录参数名。
- 等 DDP / accelerate 准备完成后，再使用稳定的模型结构。
- 只跟踪 `requires_grad=True` 的可训练参数。

因此被冻结的参数不会进入 EMA。例如 phase-1 冻结的 ViT 不会被 EMA 跟踪；phase-2 中只有 LoRA 等可训练参数会被 EMA 跟踪。

## 5. 每个训练 step 如何更新 EMA

LeRobot 的通用训练循环在每次 optimizer step 后，会检查 policy 是否有 `update()` 方法：

```python
if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
    accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
```

DM0Policy 实现了 `update()`，因此每个优化器更新之后都会同步更新 EMA：

```python
def update(self):
    if self.config.ema_decay is None:
        return

    if self._ema_params is None:
        self._init_ema()

    decay = self.config.ema_decay
    with torch.no_grad():
        for name, param in self.named_parameters():
            if param.requires_grad and name in self._ema_params:
                self._ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
```

更新公式是：

```text
ema = decay * ema + (1 - decay) * current_param
```

也就是说，EMA 参数是训练过程中可训练参数的滑动平均版本。

## 6. 从一次训练 step 看 EMA 的作用过程

一次训练 step 中，EMA 的作用顺序如下：

```text
1. policy.train()
2. forward(batch) 计算 loss
3. accelerator.backward(loss) 反向传播
4. optimizer.step() 更新真实训练参数 w_train
5. optimizer.zero_grad() 清空梯度
6. lr_scheduler.step() 更新学习率
7. policy.update() 更新 EMA 参数 w_ema
```

关键点是：EMA 发生在 `optimizer.step()` 之后。也就是说，本 step 的梯度先改变真实参数，然后 EMA 再吸收这一次更新后的参数值。

更具体地说，假设某个参数在 step `t-1` 后是：

```text
w_train = 10.0
w_ema   = 9.8
```

当前 batch 反向传播后，optimizer 把训练参数更新成：

```text
w_train = 10.4
```

如果 `decay=0.999`，EMA 更新为：

```text
w_ema = 0.999 * 9.8 + 0.001 * 10.4
      = 9.8006
```

可以看到，训练参数从 `10.0` 跳到 `10.4`，但 EMA 只从 `9.8` 变成 `9.8006`。这就是 EMA 平滑的来源。

如果很多 step 都持续把参数往同一个方向推，EMA 也会慢慢跟上；如果只是某一个 batch 造成了短暂跳动，EMA 受到的影响会很小。

### 为什么只跟踪可训练参数

代码里 `_init_ema()` 和 `update()` 都检查 `param.requires_grad`。这样做是为了让 EMA 的语义和当前训练阶段一致：

- phase-1 冻结 ViT 时，ViT 参数不会被 optimizer 改，也不需要 EMA。
- phase-2 LoRA 时，base 参数冻结，EMA 只跟踪 LoRA adapter 等可训练参数。
- full_train 时，所有打开训练的参数都会进入 EMA。

这避免了为大量冻结参数额外保存一份 shadow copy，也避免了 checkpoint 中出现“不参与训练但被 EMA 管理”的状态。

### EMA 不会影响训练梯度

EMA 参数 `_ema_params` 是用 `param.data.clone()` 创建，并在 `torch.no_grad()` 里更新：

```python
with torch.no_grad():
    self._ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
```

所以 `_ema_params` 不在 autograd 图里，不会产生梯度，也不会被 optimizer 更新。训练 loss、backward、gradient clipping、optimizer step 都仍然只围绕真实模型参数进行。

## 7. 推理时如何使用 EMA

DM0 的推理入口通常是 `select_action()`。在 `select_action()` 中，会先把当前模型参数临时替换成 EMA 参数：

```python
backup = self._swap_to_ema()
try:
    ...
finally:
    self._restore_from_backup(backup)
```

流程是：

1. `_swap_to_ema()` 备份当前训练参数。
2. 把已维护的 EMA 参数 copy 到模型里。
3. 执行 action chunk 推理。
4. `_restore_from_backup()` 恢复原始训练参数。

因此：

- 正常通过 `select_action()` 推理时，会使用 EMA 权重。
- 如果直接调用 `predict_action_chunk()`，不会自动切到 EMA 权重。
- `_ema_active` 用来避免嵌套调用时重复 swap。

### 推理时为什么要临时 swap

当前实现没有维护第二个完整的 `DM0Policy` 副本，而是只维护 `_ema_params` 这份参数字典。推理时要让已有 forward / inference 代码走 EMA 权重，最简单的方式就是：

```text
当前模型参数 w_train
    |
    | 备份 w_train
    v
把模型参数替换成 w_ema
    |
    | 调用原来的 predict_action_chunk()
    v
恢复 w_train
```

这样不用复制整个模型，也不用改 DM0 底层 `inference_action()` 的接口。代价是推理期间会有一次参数 copy，但逻辑清晰，且能保证训练参数不会被永久覆盖。

`_ema_active` 是一个保护开关。如果 `select_action()` 内部或外部发生嵌套调用，已经 swap 到 EMA 后不会再次重复 swap，避免备份和恢复顺序错乱。

## 8. Checkpoint 保存策略

保存 checkpoint 时，`lerobot/src/lerobot/utils/train_utils.py` 会先把 EMA 参数 swap 到模型里，再调用 `policy.save_pretrained()`：

```python
ema_backup = None
if hasattr(policy, "_swap_to_ema") and hasattr(policy, "_ema_params") and policy._ema_params is not None:
    ema_backup = policy._swap_to_ema()

try:
    policy.save_pretrained(pretrained_dir)
finally:
    if ema_backup is not None and hasattr(policy, "_restore_from_backup"):
        policy._restore_from_backup(ema_backup)
```

因此保存出来的：

```text
checkpoints/<step>/pretrained_model/model.safetensors
```

默认包含 EMA 权重，适合直接用于推理。

为了支持继续训练，训练状态里还会额外保存：

```text
checkpoints/<step>/training_state/ema_state.safetensors
checkpoints/<step>/training_state/raw_trainable_params.safetensors
```

含义分别是：

- `ema_state.safetensors`：EMA 影子参数。
- `raw_trainable_params.safetensors`：未替换成 EMA 的原始训练参数。

这样 resume 时可以继续用 raw 参数训练，同时恢复 EMA 状态。

### 为什么 `model.safetensors` 保存 EMA 而不是 raw 参数

训练结束后最常见的用途是推理和部署，而启用 EMA 的目的就是希望推理使用更稳定的 EMA 权重。因此 checkpoint 的 `pretrained_model/model.safetensors` 直接保存 EMA 版本，加载 `pretrained_model/` 做推理时不需要再额外找 `ema_state.safetensors`。

但继续训练时不能只依赖 `model.safetensors`，因为那里面已经是 EMA 权重，不是 optimizer 刚刚更新出来的真实训练轨迹。为了 resume 不改变训练动力学，代码额外保存 `raw_trainable_params.safetensors`，恢复时再把 raw 参数放回模型。

可以理解为 checkpoint 同时服务两种场景：

- 推理 / 部署：用 `pretrained_model/`，默认拿到 EMA 权重。
- 继续训练：用完整 checkpoint，恢复 raw 参数、optimizer、scheduler、EMA shadow state。

## 9. Resume 时如何恢复

当 `cfg.resume=True` 时，训练脚本会加载 optimizer / scheduler / rng 等训练状态，然后额外处理 EMA：

1. 如果存在 `raw_trainable_params.safetensors`，先把模型中的可训练参数恢复成 raw training params。
2. 如果存在 `ema_state.safetensors`，再恢复 `_ema_params`。

这样可以避免一个问题：`model.safetensors` 保存的是 EMA 权重，但继续训练应该从 raw training params 继续，而不是从 EMA 权重继续。

## 10. 总体调用链

完整链路如下：

```text
train_dm0_r1_pro.py
  --ema-decay
      |
      v
DM0Config.ema_decay
      |
      v
DM0Policy.update()
  每个 optimizer step 后更新 _ema_params
      |
      +--> select_action()
      |      推理前临时 swap 到 EMA 权重
      |
      +--> save_checkpoint()
             保存 pretrained_model 时使用 EMA 权重
             training_state 中另存 EMA 和 raw 参数
```

## 11. 注意事项

- 不传 `--ema-decay` 时，EMA 完全关闭。
- EMA 只覆盖当前可训练参数，不覆盖被冻结的参数。
- phase-2 LoRA 训练时，EMA 跟踪的是 LoRA 等可训练参数。
- checkpoint 中的 `pretrained_model/model.safetensors` 是面向推理的 EMA 权重。
- resume 训练时依赖 `training_state/raw_trainable_params.safetensors` 恢复原始训练参数。
- 如果只拷贝 `pretrained_model/`，可以用于推理；如果要继续训练，应该保留完整 checkpoint 目录。

## 12. 推荐用法

一般训练可以这样启用：

```bash
--ema-decay=0.999
```

如果训练 step 很多、希望参数更平滑，可以尝试：

```bash
--ema-decay=0.9999
```

如果只是快速 debug 或短跑 dry-run，可以不启用 EMA，保持默认 `None`。
