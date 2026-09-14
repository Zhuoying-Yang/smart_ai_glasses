# WHEN — 决定「什么时候该叫模型」

计算路由系统的 WHEN 层。它持续看画面、听语音，只在值得的时候产出一条标签交给下游的
WHICH router。绝大多数时间它什么都不做——这正是它存在的意义。

**零训练**：视觉判断用 SigLIP 的 zero-shot 图文相似度，不训练任何参数。

三条通路：

| 类型 | 触发源 | 说明 |
|---|---|---|
| `INSTANT` | 麦克风 | 你开口提问，不经过视觉门，直接放行 |
| `STANDING` | 摄像头 | 你注册的持续监测任务（「拿起手机就提醒我」）|
| `ALERT` | 摄像头 | 系统预置的预警（跌倒、起火），本质是一条系统给定的 standing query |

---

## 1. 环境安装

需要 **Python 3.11**。语音功能目前只在 Apple Silicon 上验证过（依赖 `mlx-whisper`）。

### 方式 A：conda

```bash
conda create -n eyewhen python=3.11 -y
conda activate eyewhen
pip install -r when/requirements.txt
```

### 方式 B：uv（更快）

```bash
uv venv --python 3.11
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r when/requirements.txt
```

### 只做视频分析、不用语音

`requirements.txt` 里 `sounddevice` 和 `mlx-whisper` 两行可以删掉，其余都是必需的。

### 验证

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Apple Silicon 上应输出 `True`（用 Metal 加速）。没有 GPU 也能跑，会退到 CPU，慢一些。

> **所有命令都在仓库根目录执行**（`when/` 的上一层），因为入口是 `python -m when.xxx`。

---

## 2. 快速开始

```bash
# 分析一段视频，边放边出标签
python -m when.run_video path/to/video.mov

# 开摄像头 + 麦克风，说话注册监测任务
python -m when.run_live --no-preset --negatives auto

# 语音进不去时先跑这个（实时音量表）
python -m when.run_live --mic-test
```

---

## 3. `run_video` — 视频分析

```bash
python -m when.run_video <视频路径> [参数]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `<视频路径>` | 必填 | 任何 OpenCV 能解码的文件 |
| `--deterministic` | 关 | **离线评测必须加**。按帧号采样、不丢帧、不按墙上时间播放。不加的话门跑在后台线程、忙时会丢帧，同一视频两次跑结果不同 |
| `--no-display` | 关 | 不开播放窗口，只输出标签。跑批量分析时用 |
| `--speed N` | `1.0` | 播放倍速。`--speed 8` 快速过一遍 |
| `--jsonl 文件` | 无 | 把完整事件流（含 SILENT 和连续分数）写成 JSONL，后续分析用 |
| `--negatives {off,manual,auto}` | 读配置 | 覆盖负样本模式，见第 5 节 |
| `--model 名称` | 读配置 | 换 SigLIP 模型，如 `google/siglip-so400m-patch14-384` |
| `--fps N` | `2.0` | 视觉门采样率 |
| `--min-raw N` | 读配置 | 覆盖绝对下限。**标定时设 `0`** 可以看到全部原始分数 |
| `--only id1 id2` | 全部 | 只启用指定的 query id |
| `--config 路径` | `when/queries.yaml` | 换一份配置文件 |

**常用组合**

```bash
# 看效果：开窗口实时播放
python -m when.run_video clips/desk.mov

# 出可复现的结果
python -m when.run_video clips/desk.mov --deterministic --no-display

# 标定阈值：关掉下限，导出全部分数
python -m when.run_video clips/desk.mov --deterministic --no-display --min-raw 0 --jsonl scores.jsonl

# 对比两个模型
python -m when.run_video clips/desk.mov --deterministic --no-display --model google/siglip-so400m-patch14-384
```

---

## 4. `run_live` — 摄像头 + 语音

```bash
python -m when.run_live [参数]
```

### 诊断类（跑完即退出）

| 参数 | 说明 |
|---|---|
| `--list-devices` | 列出所有麦克风和摄像头的编号 |
| `--mic-test` | 实时音量表。说话时条形要冲过 `│` 那道阈值线。**说话识别不出来时先跑这个** |

### 运行类

| 参数 | 默认 | 说明 |
|---|---|---|
| `--source {mac,aria}` | 询问 | `mac`=电脑摄像头，`aria`=眼镜。**都不给且在终端里跑时会让你选 1 或 2**，选眼镜会自动拉起桥接 |
| `--no-preset` | 关 | 不加载配置里的 standing/alerts，**开机零 query，只监测你口头注册的**。笔记本摄像头对着自己时建议开 |
| `--negatives {off,manual,auto}` | 读配置 | 新环境建议 `auto`，开头 4 秒自动探测环境 |
| `--camera N` | 见 `--source` | 摄像头编号或 MJPEG/RTSP 地址。给了它就不再询问来源 |
| `--aria-python` | `~/aria_env/bin/python` | 装了 Aria SDK 的解释器 |
| `--aria-port N` | `8080` | 桥接端口。端口已被占用时直接复用，不会起第二个 |
| `--aria-size N` | `640` | 桥接下采样到的边长 |
| `--mic N` | 系统默认 | 麦克风编号，用 `--list-devices` 查 |
| `--no-audio` | 关 | 只跑视觉，不开麦克风 |
| `--no-display` | 关 | 不开预览窗口 |
| `--jsonl 文件` | 无 | 导出完整事件流 |
| `--fps N` | `2.0` | 视觉门采样率 |
| `--only id1 id2` | 全部 | 只启用指定 query id |
| `--config 路径` | `when/queries.yaml` | 换配置文件 |
| `--model 名称` | 读配置 | 换 SigLIP 模型 |

### 语音相关

| 参数 | 默认 | 说明 |
|---|---|---|
| `--asr-model 名称` | `mlx-community/whisper-small-mlx` | `small` 又快又支持中译英；`medium` 中文准确率更高但慢约 2.7 倍。**`large-v3-turbo` 不支持翻译**，说中文会得到中文 query |
| `--language zh\|en` | 自动判别 | 指定语种可以略微加速 |
| `--no-translate` | 关 | 不翻成英文。**不建议**——SigLIP 文本塔是英文训练的，中文 query 匹配不上 |
| `--vad-threshold N` | `0.006` | VAD 能量地板值。启动时会按环境底噪自动抬高（底噪 × 2），这个值只是下限。环境很吵、说话进不去时调大 |

### 说话能做什么

启动后对着麦克风说（中英文都行）：

| 你说 | 效果 |
|---|---|
| 「如果我拿起手机就提醒我」 | 注册一条 STANDING，之后持续监测 |
| 「这是什么」 | 判为 INSTANT，立刻出一条标签 |
| 「我在监测什么」 | 列出当前所有任务及各自阈值 |
| 「全部取消」 | 清空所有已注册的任务 |

口头指令会被自动剥成画面描述再喂给 SigLIP：

```
当我拿起手机的时候告诉我
  → When I pick up my phone, tell me      （英译）
  → a hand holding a phone                （喂给 SigLIP 的描述）
```

**注册后留意打印出来的「匹配用描述」**。翻译偶尔会出错，描述不对就说「全部取消」重说一次。

### 常用组合

```bash
# 推荐：零 query 起步，环境自动探测
python -m when.run_live --no-preset --negatives auto

# 中文准确率优先（慢约 1.5 秒）
python -m when.run_live --no-preset --negatives auto --asr-model mlx-community/whisper-medium-mlx

# 只测视觉，不开麦克风
python -m when.run_live --no-audio --negatives auto

# 环境吵，手动抬高语音门槛
python -m when.run_live --no-preset --negatives auto --vad-threshold 0.03
```

---

## 4.5 用 Project Aria 眼镜当输入

**Aria 不会注册成摄像头。** 它不实现 USB Video Class，系统摄像头列表里看不到它，
`cv2.VideoCapture(1)` 也找不到——图像只能通过 Aria Client SDK 的流式接口取。

而 SDK 的依赖树跟本项目冲突（会拖进 jupyter / matplotlib / rerun 等 130 多个包，
还会降级 pillow），所以它必须装在**独立的环境**里。

`when/aria_bridge.py` 就是那座桥：在 Aria 环境里取流，转成本地 MJPEG，
我们这边用 `cv2.VideoCapture(url)` 原生读取。

### 一次性准备

```bash
python3 -m venv ~/aria_env
source ~/aria_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install projectaria_client_sdk --no-cache-dir
python3 -m pip install opencv-python

aria auth pair
aria auth check
aria device status
```

`aria auth pair` 时眼镜上会弹确认，需要在眼镜上点一下。
**命令后面不要跟注释**——交互式 zsh 默认不把 `#` 当注释，会把注释当参数传进去。

### 跑起来

**方式一（推荐）：一条命令，`run_live` 自动拉起桥接**

```bash
python -m when.run_live --no-preset --negatives auto
```

不带 `--source` 时会先问你：

```
画面来源：
  1) 电脑摄像头
  2) 眼镜摄像头（Project Aria）
选择 [1/2]:
```

选 2 就会自动启动桥接（约 15 秒）、连上、退出时自动收尾。想跳过询问就直接
`--source aria` 或 `--source mac`。

**方式二：两个终端，桥接自己管**

适合桥接要长时间开着、反复重启 `run_live` 的场景。已经有桥接在跑时，
方式一会直接复用它，不会起第二个——眼镜同一时间只允许一个流式会话。

```bash
# 终端 1 —— Aria 环境。必须用 -m，不能按路径运行（见下）
cd <仓库根目录>
~/aria_env/bin/python -m when.aria_bridge

# 终端 2 —— 本项目环境
python -m when.run_live --camera http://127.0.0.1:8080/ --no-preset --negatives auto
```

### 只想看画面，不做任何分析

```bash
~/aria_env/bin/python -m when.aria_view
```

**在终端里按 `q` 退出**（不是在窗口里）。从终端启动的 Python 不是 macOS app
bundle，它的 OpenCV 窗口拿不到键盘焦点——关闭按钮是灰的，按键会落到终端上。
所以脚本改成在终端收键。

### `aria_bridge` 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--port N` | `8080` | MJPEG 服务端口 |
| `--host` | `127.0.0.1` | 默认只监听本机，不暴露到网络 |
| `--profile` | `profile12` | `profile12` = RGB 10fps 2MP 无音频；`profile18` 带空间音频 |
| `--interface {usb,wifi}` | `usb` | USB 延迟更低更稳 |
| `--size N` | `640` | 把 1408×1408 缩到这个尺寸再编码；`0` 保持原尺寸 |
| `--quality N` | `80` | JPEG 质量 |
| `--no-rotate` | 关 | 跳过 90° 旋转校正 |
| `--serial` | 自动 | 接了多台设备时指定序列号 |

### 三个必须知道的坑

**1. 原始帧是躺倒的。** Aria 的 RGB 传感器是旋转安装的，直接取到的帧转了 90°。
桥接默认做 `np.rot90(frame, -1)` 校正——这是拿已知正立的物体实测出来的，不是猜的。
不校正的话画面是横的，SigLIP 什么都匹配不上。

**2. 必须用 `-m` 启动桥接。** 按路径运行（`python when/aria_bridge.py`）会把 `when/`
放进 `sys.path[0]`，于是 `when/types.py` 遮蔽标准库的 `types`，导致标准库深处循环导入崩溃。
脚本里加了守卫，误用时会直接给出提示。

**3. 同一时间只能有一个流式会话。** 桥接被强杀（没跑完 `stop_streaming`）后，
眼镜端会一直以为会话还开着，下次启动报 `(940) Cannot start streaming...`。解法：

```bash
~/aria_env/bin/aria streaming stop
```

`run_live` 正常退出时会给桥接发 SIGINT 让它自己收尾，不会留下残留。

**4. `~/aria_env` 不要在激活 conda 环境时创建。** venv 会继承创建时那个解释器的标准库。
影响不大，但混淆调试。

### 实测数据

| 项 | 数值 |
|---|---|
| 原始流 | 1408×1408 RGB，9.9 fps（profile12）|
| 经桥接（`--size 640`）| 640×640，10.5 fps |
| 门实际消耗 | 2 fps（其余帧直接丢弃）|

---

## 5. 配置：`when/queries.yaml`

所有行为都在这里改，不用动代码。

### 负样本模式 `negatives.mode`

负样本是「什么算正常」的参照物。系统问的不是"这像不像水瓶"，而是"这更像水瓶还是更像我列的那些正常东西"。

| 模式 | 说明 | 什么时候用 |
|---|---|---|
| `off` | 不用负样本，纯余弦。零配置，换任何环境直接跑，略糙 | 陌生环境快速试 |
| `manual` | 自己写几句。最准，但换环境要重写 | 固定场景 |
| `auto` | 开头 4 秒探测，从 174 条通用词表里自动挑 6 条 | 换环境不用配置，但这两段视频上仍略逊于 manual |

`auto` 挑选时有两道约束：**类别限额**（场景/物体/人/画质每类最多 2 条，防止 6 条
全挤在一个语义簇里）和 MMR（默认关闭——三个场景实测它只改变过顺序、没改变过集合）。

### 触发条件（两个都要满足）

```
lift    = 快 EMA − 慢基线     回答「刚刚变了没」
min_raw = 绝对分数下限        回答「东西真的在画面里没」
```

只看 `lift` 会被「物体一直在画面里」骗到；只看 `min_raw` 分不出「一直在」和「刚出现」。

阈值在启动探测期**逐条 query 自动标定**，锚定在 softmax 的随机水平 `1/(1+K)`（K = 负样本条数）。低于随机水平说明模型根本没匹配上。

### 常调的几个值

| 键 | 默认 | 调它干嘛 |
|---|---|---|
| `gate.fps` | `2.0` | 采样率。调高更灵敏也更耗电 |
| `gate.cooldown_s` | `20.0` | 同一条 query 触发后的静默期。**反复做同一个动作测试时要调小到 5** |
| `negatives.auto.chance_multiplier` | `1.2` | 乱触发就调大，该触发不触发就调小 |
| `negatives.auto.probe_seconds` | `4.0` | 开头探测环境的时长 |

### 自己写 query

写**画面长什么样**，不要写祈使句：

```yaml
standing:
  - id: sq_bottle
    text: a water bottle placed close to a keyboard   # ✅
    urgency: normal
  # ❌ warn me if I put the bottle near the keyboard
```

**别写复合条件**（「A 且 B」）。SigLIP 学的是「图里有什么」，不是「A 和 B 什么关系」，
实测更大的模型在这类 query 上判别方向反而是错的。

---

## 6. 输出格式

终端每个采样点刷一行状态条，触发时另起高亮行：

```
[   9.50s] al_fall:········+0.00  sq_bottle:██████··+0.06
[  11.50s] ▶ TRIGGER STANDING  sq_bottle  lift=+0.114 thr=0.10 (raw=0.771 base=0.595) nov=0.170  urgency=normal  frames=[19, 20, 21, 22, 23]
           query: a water bottle placed close to a keyboard
```

- **条形** = 绝对相似度 `raw`（画面里有没有这东西）
- **数字** = `lift`（存在感刚刚有没有变强）—— **触发只看这个**

`--jsonl` 导出的每一行是一个完整事件，这就是交给 WHICH router 的东西：

```json
{ "seq": 106, "t_emit": 11.501, "route": "TRIGGER", "trigger_type": "STANDING",
  "query": {"text": "a water bottle placed close to a keyboard",
            "origin": "user_standing", "query_id": "sq_bottle"},
  "score": 0.1029, "threshold": 0.1, "raw_score": 0.7715, "baseline": 0.6017,
  "evidence": {"window": [9.501, 11.501], "frame_idx": [19,20,21,22,23], "novelty": 0.1698},
  "urgency": "normal", "cooldown_until": 16.501 }
```

`evidence.frame_idx` 就是给下游的「该看哪几帧」。`route` 为 `SILENT` 的行也会输出——
既是心跳，也保留了连续分数供后续标定阈值。

---

## 7. 排查

| 现象 | 怎么办 |
|---|---|
| 卡在「加载 SigLIP ...」不动 | 网络问题。模型已缓存时**不该再联网**，现在是缓存优先；若仍卡住，说明这个模型没下载过，只能等它下完或换网 |
| `ModuleNotFoundError: No module named 'torch'` | 用错解释器了。提示符里若同时有 `(aria_env)`，先 `deactivate`。**本项目从不需要激活 `aria_env`**——Aria 相关命令都用绝对路径 `~/aria_env/bin/...` |
| 说话没反应 | `python -m when.run_live --mic-test`，看条形有没有冲过阈值线 |
| 摄像头打不开 | macOS：系统设置 → 隐私与安全性 → 摄像头，勾上终端/VSCode，**然后完全退出该程序再重开** |
| 识别出 `Thank you for watching` 之类 | Whisper 在静音上的幻觉，已有黑名单过滤。频繁出现说明 VAD 阈值太低 |
| 说中文但 query 是中文 | ASR 模型不支持翻译。换 `--asr-model mlx-community/whisper-small-mlx` |
| 同一动作重复做不触发 | 冷却期（默认 20 秒）。改 `queries.yaml` 的 `gate.cooldown_s` |
| 同一视频两次跑结果不同 | 离线分析要加 `--deterministic` |

启动时的三条自检信息能定位大部分问题：

```
✓ 摄像头 0 正常出帧
✓ 麦克风 [MacBook Air麦克风] 已打开
  正在测环境底噪（1.5 秒，请保持安静）... 底噪=0.0049  触发阈值=0.0098
```

---

## 8. 性能参考

Apple M4（MacBook Air，16GB，MPS）实测：

| 项 | 数值 |
|---|---|
| SigLIP-base-224 单帧 | 约 40–70 ms（上限 14–24 FPS，实际只需 2 FPS）|
| SigLIP-so400m-384 单帧 | 约 265 ms（慢约 6 倍）|
| 常驻内存 | 约 2.6 GB（SigLIP + Whisper）|
| 语音端到端（中文，small）| 约 1.5 秒（VAD 挂起 0.7s + 转写 0.4s + 英译 0.4s）|
| 语音端到端（中文，medium）| 约 3.0 秒 |

ASR 耗时几乎与说话长短无关——Whisper 内部把任何输入补齐到 30 秒，成本按次算不按秒算。

---

## 9. 相关文档

- [GIT_WORKFLOW.md](GIT_WORKFLOW.md) — 版本管理流程与回滚方法
