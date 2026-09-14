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
| `--no-preset` | 关 | 不加载配置里的 standing/alerts，**开机零 query，只监测你口头注册的**。笔记本摄像头对着自己时建议开 |
| `--negatives {off,manual,auto}` | 读配置 | 新环境建议 `auto`，开头 4 秒自动探测环境 |
| `--camera N` | `0` | 摄像头编号 |
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

## 5. 配置：`when/queries.yaml`

所有行为都在这里改，不用动代码。

### 负样本模式 `negatives.mode`

负样本是「什么算正常」的参照物。系统问的不是"这像不像水瓶"，而是"这更像水瓶还是更像我列的那些正常东西"。

| 模式 | 说明 | 什么时候用 |
|---|---|---|
| `off` | 不用负样本，纯余弦。零配置，换任何环境直接跑，略糙 | 陌生环境快速试 |
| `manual` | 自己写几句。最准，但换环境要重写 | 固定场景 |
| `auto` | 开头 4 秒探测，从 112 条通用词表里自动挑 6 条 | **推荐**，兼顾两者 |

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
