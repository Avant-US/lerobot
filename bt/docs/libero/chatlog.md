
# `Do you want to specify a custom path for the Libero
**User**

When I run lerobot-eval, it shows `Do you want to specify a custom path for the dataset folder? (Y/N):`. What is this custom path using for?

---

**Cursor**

正在代码库中查找该提示的来源及用途。



这条提示**不是** `lerobot-eval` 自己打印的，而是 **`libero` 包在第一次被加载时**触发的（例如你用 `lerobot-eval` 跑 **LIBERO** 环境时会 `import libero`）。

## 它在问什么？

在 **`~/.libero/config.yaml` 还不存在**时，`libero` 会初始化配置，并问你是否要**自定义「数据集目录」的路径**。

对应代码在已安装的 `libero` 里（逻辑如下）：

```100:129:/home/Luogang/SRC/Robot/lerobot/lerobot-venv/lib/python3.12/site-packages/libero/libero/__init__.py
if not os.path.exists(config_file):
    # Create a default config file

    default_path_dict = get_default_path_dict()
    answer = input(
        "Do you want to specify a custom path for the dataset folder? (Y/N): "
    ).lower()
    if answer == "y":
        # If the user wants to specify a custom storage path, prompt them to enter it
        custom_dataset_path = input(
            "Enter the path where you want to store the datasets: "
        )
        full_custom_dataset_path = os.path.join(
            os.path.abspath(os.path.expanduser(custom_dataset_path)), "datasets"
        )
        ...
            default_path_dict["datasets"] = full_custom_dataset_path
    ...
    with open(config_file, "w") as f:
        yaml.dump(default_path_dict, f)
```

## 这个「自定义路径」用来干什么？

- **用途**：指定 **LIBERO 相关数据集文件** 存放在磁盘上的**根目录**（最终会写成 `你输入的路径/datasets`，并写入配置里的 `datasets` 这一项）。
- **默认**：若选 **N**，会用包内推导的默认位置（`get_default_path_dict()` 里相对于 libero 安装目录的 `../datasets` 等一整套路径）。
- **为何需要自定义**：常见场景是默认盘空间不够，想把大数据集放到**另一块盘或 NFS 路径**。

配置目录默认是 **`~/.libero`**（也可用环境变量 **`LIBERO_CONFIG_PATH`** 改掉），里面的 **`config.yaml`** 会记录 `benchmark_root`、`bddl_files`、`init_states`、`datasets`、`assets` 等路径，之后 LIBERO 会按这个文件找资源。

**简要结论**：选 **Y** 时，你在指定 **LIBERO 的 `datasets` 目录放哪儿**；和 LeRobot 自己的 `LeRobotDataset` 训练数据路径不是一回事，只是 LIBERO 仿真/基准用的数据落盘位置。


## 能否在使用 lerrobot-eval时,在命令行就对该 custom path 进行设置, 而不用在运行时跳出提示让人设置.
---

**User**

能否在使用 lerrobot-eval时,在命令行就对该 custom path 进行设置, 而不用在运行时跳出提示让人设置.

---

**Cursor**


查明 LIBERO 的初始化时机及 LeRobot 的集成方式，以便支持通过 CLI 或环境变量避免交互式提示。




在导入 `libero.libero` 之前写入 `config.yaml`，可跳过交互提示。正在实现 `ensure_libero_config`，并在 `factory` 中于导入前调用；添加 `LiberoEnv.libero_dataset_path` 与环境变量 `LEROBOT_LIBERO_DATASET_PATH` 支持。




正在添加 `libero_config.py`，并更新 `factory.py` 与 `LiberoEnv` 配置。




已在仓库里接上**非交互**的 LIBERO 配置逻辑，用法如下。

## 原因,为什么会问该路径

交互提示来自 `libero` 在**首次**导入时：若 `LIBERO_CONFIG_PATH`（默认 `~/.libero`）下还没有 `config.yaml`，就会在 `stdin` 上提问。  
现在在 **`import libero.libero` 之前**先写好 `config.yaml`，就不会再弹问。

## 用法,如何提前设置该路径

**1. 命令行（推荐）** — 与 `datasets` 在 LIBERO 配置里一致，填**数据集目录本身**（不是父目录；若选自定义目录，等价于 interactive 里最终用的 `.../datasets` 路径）：

```bash
lerobot-eval \
  --env.type=libero \
  --env.libero_dataset_path=/你的路径/datasets \
  ...其它参数...
```

**2. 环境变量**（不设 `--env.libero_dataset_path` 时生效）：

```bash
export LEROBOT_LIBERO_DATASET_PATH=/你的路径/datasets
lerobot-eval --env.type=libero ...
```

优先级：**`--env.libero_dataset_path` > `LEROBOT_LIBERO_DATASET_PATH`**。

**3. 不改路径、只要非交互**  
不传上面两项也可以：首次会在 `~/.libero/config.yaml` 写入与官方默认一致的相对路径（等价于交互里选 **N**），同样不会再问。

**4. 配置文件所在目录**  
仍由上游的 `LIBERO_CONFIG_PATH` 控制（默认 `~/.libero`），与原来一致。

---

实现要点：新增 `src/lerobot/envs/libero_config.py` 的 `ensure_libero_config()`，并在 `factory.py` 里在 `from lerobot.envs.libero import create_libero_envs` **之前**调用；`LiberoEnv` 增加字段 `libero_dataset_path`（见 `configs.py`）。若你本地 `configs.py` 里还没有该字段，保存/拉取后应能看到 `libero_dataset_path: str | None = None` 一段。
---


# LIBERO 官方数据集与HF的数据集

**User**

libero 下载的 dataset 是从哪里下载的? 和 Huggingface的 https://huggingface.co/datasets/HuggingFaceVLA/libero  上的 HuggingFaceVLA/libero 数据集有什么不同和相同的地方

---

**Cursor**


正在查看 `libero` 包中数据集的下载来源与用途。




根据你环境里安装的 `libero` 包源码，可以这样理解「官方 LIBERO 下载的数据」和 [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero) 的关系。

## LIBERO 包里的 dataset 从哪下？

在 `libero/libero/utils/download_utils.py` 里，**模仿学习用的演示数据**（HDF5 等）有两种来源：

1. **默认（`use_huggingface=False`）**  
   从 **UT Austin 的 Box 网盘**拉 zip（代码里写明 “these may expire soon”）：

```101:106:/home/Luogang/SRC/Robot/lerobot/lerobot-venv/lib/python3.12/site-packages/libero/libero/utils/download_utils.py
DATASET_LINKS = {
    "libero_object": "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip",
    "libero_goal": "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip",
    "libero_spatial": "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
    "libero_100": "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
}
```

2. **可选（`use_huggingface=True`）**  
   从 Hugging Face 上的 **`yifengzhu-hf/LIBERO-datasets`** 用 `snapshot_download` 按子目录拉（例如 `libero_spatial/*`），不是 `HuggingFaceVLA/libero`。

**仿真资源（物体/场景 mesh 等）** 是另一路：从 HF **`lerobot/libero-assets`**（`repo_type="dataset"`）下载，供 MuJoCo 里跑环境用。

你在交互里配的 **`config.yaml` 里的 `datasets`**，指的是上面这类 **LIBERO 原生演示数据**要落在哪个目录（`libero_dataset_download` / `get_libero_path("datasets")` 会用），和 LeRobot 的 Hub 数据集目录不是同一个概念。

---

## 和 HuggingFaceVLA/libero 的相同点

- **任务语义上一致**：都是围绕 **LIBERO** 那套桌面操作基准的演示数据（多相机图像、状态、动作等），用于模仿学习 / VLA 训练。
- **内容来源可视为同一条研究 lineage**：HF 页上写 *“This dataset was created using LeRobot”*，相当于把同一批 benchmark 经验做成了 **LeRobot 生态**里统一可用的版本。

---

## 不同点（重点）

| 维度 | LIBERO 包下载的数据 | [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero) |
|------|---------------------|-------------------------------------------------------------------------------|
| **Hub 仓库** | `yifengzhu-hf/LIBERO-datasets`（可选）或 Box zip | `HuggingFaceVLA/libero` |
| **存储格式** | 以 **HDF5 + 原有目录结构**为主（`libero_dataset_download` / `check_libero_dataset` 里按 `*.hdf5` 检查） | **LeRobot v3**：`meta/info.json`、`parquet`、`mp4` 视频等 |
| **典型用途** | LIBERO 原版 / `libero` 包里的 **lifelong、原生 dataloader** | **`lerobot-train` 等**用 `repo_id=HuggingFaceVLA/libero` 训练 |
| **和 `lerobot-eval` 的关系** | 评估主要用 **bddl、init_states、仿真 assets**；不一定要在 eval 时读这份 HDF5 | eval **一般不直接读**这个 parquet 数据集；训练时才用 |

---

**一句话**：  
- **LIBERO 自带的下载** = 官方（或镜像到 `yifengzhu-hf/LIBERO-datasets`）的 **HDF5 演示包**，给 LIBERO 自己的训练管线用。  
- **HuggingFaceVLA/libero** = 同一类 LIBERO 数据在 Hugging Face 上 **用 LeRobot 格式重打包** 的数据集，给 LeRobot 训练用；**仓库、文件格式、调用方式都不同**，只是任务与数据含义高度重叠。

---

# LIBERO上的train和eval分别会用到什么数据/资源
**User**

以libero官方为例, train 和 evaluation 时分别用到哪些数据和资源? 这些数据和资源分别起什么作用?

---

**Cursor**


结合 LIBERO 官方包的路径设计与 LeRobot 中的用法，梳理训练与评估各自依赖的资源。




下面按 **LIBERO 官方包**（`~/.libero/config.yaml` 里那几条路径 + 下载脚本）来说明：**训练**和**评估**各用哪些东西、各自干什么。

---

## 配置里这几类资源分别是什么

官方在 `get_default_path_dict` 里把资源分成五类（你本机路径可能因安装方式略有不同）：

```14:38:/home/Luogang/SRC/Robot/lerobot/lerobot-venv/lib/python3.12/site-packages/libero/libero/__init__.py
def get_default_path_dict(custom_location=None):
    ...
    bddl_files_default_path = os.path.join(benchmark_root_path, "./bddl_files")
    init_states_default_path = os.path.join(benchmark_root_path, "./init_files")
    dataset_default_path = os.path.join(benchmark_root_path, "../datasets")
    assets_default_path = os.path.join(benchmark_root_path, "./assets")
    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": bddl_files_default_path,
        "init_states": init_states_default_path,
        "datasets": dataset_default_path,
        "assets": assets_default_path,
    }
```

---

## 训练（Train）时用到什么

| 资源 | 是否必需（典型 IL/BC） | 作用 |
|------|------------------------|------|
| **`datasets`（HDF5 演示）** | **是** | 离线专家轨迹：观测、动作、任务标签等，供行为克隆 / 序列模型学习策略。**通过 `libero_dataset_download` 下到 `get_libero_path("datasets")`（Box 或 `yifengzhu-hf/LIBERO-datasets`）。** |
| **`bddl_files`** | 间接相关 | 定义「任务语义」（物体、目标关系）。训练若**只读 HDF5**不一定每 step 都解析 BDDL，但数据本身是在这些任务设定下采集的。 |
| **`init_states`** | 视设定而定 | 若训练管线里要**在仿真里 rollout、数据增强、或对齐采集时的复位分布**，会用与任务对应的初始状态文件。 |
| **`assets`** | 若在仿真里跑环境则需要 | MuJoCo 网格/纹理/场景等；**只在训练里要开仿真器**（如 DAGGER、在线微调、在环数据）时才必须。纯离线读 HDF5 时可以不加载仿真。 |
| **benchmark 代码（包内 `benchmark`）** | 常用 | 任务列表、suite 划分、任务与文件名的对应，方便按 `libero_spatial` / `libero_10` 等组织数据与实验。 |

**概括**：官方 LIBERO **训练的核心数据**是 **`datasets` 下的 HDF5 演示**；其余多为「任务定义 + 仿真世界」的支撑，是否在训练路径里出现取决于你是否在仿真里采样或增强。

---

## 评估（Evaluation）时用到什么

| 资源 | 是否必需 | 作用 |
|------|----------|------|
| **`bddl_files`** | **是** | 每个 episode 用哪个 **BDDL** 建环境：场景里有什么、成功条件等（LeRobot 里也是 `get_libero_path("bddl_files")` + 任务里的相对路径）。 |
| **`init_states`** | **几乎总是**（官方基准） | 每个 task 的 **`.pt` 初始状态集合**：复位时从里面取，保证与论文/基准可比的起点分布。LeRobot 的 `get_task_init_states` 就是读这里。 |
| **`assets`** | **是**（视觉策略在仿真里评） | 仿真里可见可碰撞的物体/场景资源；缺了 MuJoCo 环境起不来或缺模型。可从包内 `assets` 或 HF **`lerobot/libero-assets`** 拉取。 |
| **`datasets`（HDF5）** | **否** | 评估是 **当前策略在仿真里 rollout**，不依赖人类演示文件。 |
| **benchmark 定义** | **是** | 决定要跑哪些 task_id、suite 名称等。 |

**概括**：**评估 = 仿真 + 任务描述（BDDL）+ 复位分布（init_states）+ 场景资源（assets）**；**不需要**下载到 `datasets` 的那份 HDF5。

---

## 和 LeRobot 的对应关系（便于对照）

- **`lerobot-eval --env.type=libero`**：走的是 **评估** 这条链 → 主要用 **BDDL、init_states、assets**（以及包内任务逻辑），**不用** `HuggingFaceVLA/libero` 那种 parquet，也**不必**有官方 HDF5 `datasets`。
- **`lerobot-train` + `repo_id=HuggingFaceVLA/libero`**：走的是 **训练** 这条链 → 用 **Hub 上的 LeRobot 格式数据**；与官方 `datasets` 目录里的 HDF5 **不是同一套文件**，但任务语义上同属 LIBERO 类数据。

---

## 一句话对照

- **Train**：主要靠 **`datasets` 里的演示（HDF5）** 学策略；仿真相关资源（**assets / bddl / init_states**）只在「要在仿真里交互或对齐分布」时需要。  
- **Eval**：只靠 **仿真世界**：**assets** 建场景、**bddl** 定任务、**init_states** 定复位；**不用**演示数据集 HDF5。
---
