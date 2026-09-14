# 架构图

四张 Mermaid 图。把代码块里的内容粘进 [mermaid.live](https://mermaid.live) 或
MermaidEditor.io 即可渲染。

**实线 = 已实现，虚线框 = 未实现。**

---

## 1. 总览：WHEN 与 WHICH 的分工

```mermaid
flowchart TB
    subgraph IN["输入"]
        CAM["摄像头 / Aria 眼镜 / 视频文件"]
        MIC["麦克风"]
    end

    subgraph WHEN["WHEN 层 —— 决定「现在该不该叫模型」（已实现）"]
        direction TB
        PROBE["启动探测 4s<br/>定负样本 + 标定逐条阈值"]
        GATE["视觉门<br/>SigLIP zero-shot，2 FPS"]
        VOICE["语音通路<br/>VAD → Whisper → 意图判别"]
        PROBE -.->|"负样本 + 阈值"| GATE
        VOICE -->|"注册 / 注销 standing query"| GATE
    end

    EVT{{"WhenEvent<br/>route · trigger_type · query<br/>score / threshold / raw / baseline<br/>evidence · urgency · cooldown_until"}}

    subgraph WHICH["WHICH 层 —— 决定「交给谁、花多少算力」（未实现）"]
        direction TB
        DEC["统一决策函数<br/>共享 encoder + 每动作一个 sigmoid 头"]
        ACT["动作空间<br/>SKIP / MEMORY_ONLY<br/>SMALL×帧数×分辨率<br/>LARGE×帧数"]
        FB["fallback 检查<br/>弃权信号 + 平均 logprob"]
        DEC --> ACT --> FB
    end

    OUT["语音答复"]
    MEM[("记忆系统")]
    COST["成本模型<br/>延迟 + 能耗 + 上行字节"]

    CAM --> GATE
    MIC --> VOICE
    GATE -->|"route=TRIGGER"| EVT
    GATE -.->|"route=SILENT（心跳 + 连续分数留档）"| EVT
    VOICE -->|"INSTANT 直接放行"| EVT
    EVT --> DEC
    MEM -.->|"命中分"| DEC
    COST -.-> DEC
    FB --> OUT
    FB -.->|"置信度低 → 升级"| ACT

    classDef done fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef todo fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray:5 4,color:#616161
    classDef contract fill:#fff8e1,stroke:#f9a825,color:#e65100
    class PROBE,GATE,VOICE done
    class DEC,ACT,FB,MEM,COST todo
    class EVT contract
```

WHEN 的职责边界：**只回答「该不该叫」，不回答「发生了什么」**。
分辨「拿起眼镜」还是「戴上眼镜」是 WHICH 的活——它那边才有真正能理解画面的大模型。

---

## 2. 视觉门：每一帧的判定链路

```mermaid
flowchart TB
    F["一帧 RGB"] --> ENC["SigLIP 编码<br/>约 50 ms"]
    ENC --> NOV["novelty = 1 − cos 相邻帧<br/>（Dispider 的场景切段信号）"]
    NOV --> RP{"novelty > 0.35 ?"}
    RP -->|"是，场景换了"| REPROBE["重新探测"]
    RP -->|"否"| SC

    SC["逐 query 打分"] --> MODE{"有负样本？"}
    MODE -->|"有"| SM["softmax over<br/>[该 query, 负样本×K]"]
    MODE -->|"无"| COS["直接用余弦<br/>零配置，略糙"]
    SM --> EMA
    COS --> EMA

    EMA["双 EMA（带 bias correction）<br/>fast α=0.35　slow α=0.03<br/>lift = fast − slow"]

    EMA --> C1{"warm ?<br/>t ≥ 1.5s"}
    C1 -->|"否"| SILENT
    C1 -->|"是"| C2{"present ?<br/>raw ≥ min_raw"}
    C2 -->|"否"| SILENT
    C2 -->|"是"| C3{"lift ≥ delta_on ?"}
    C3 -->|"否"| SILENT
    C3 -->|"是"| C4{"过了冷却期 ?"}
    C4 -->|"否"| SILENT
    C4 -->|"是"| FIRE

    FIRE["▶ TRIGGER"] --> SNAP["slow ← fast<br/>（报过了，新画面从此算正常）"]
    SNAP --> LATCH["锁住，直到 raw < min_raw×0.9<br/>才重新武装"]
    LATCH --> EV
    SILENT["route = SILENT"] --> EV
    EV["WhenEvent"]

    classDef gate fill:#ffebee,stroke:#c62828,color:#b71c1c
    classDef calc fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    class C1,C2,C3,C4 gate
    class ENC,NOV,SM,COS,EMA calc
```

**两个条件缺一不可**：

| 条件 | 回答什么 | 少了它会怎样 |
|---|---|---|
| `lift ≥ delta_on` | 刚刚**变了**没 | 只看 raw：物体一直在画面里就会一直报 |
| `raw ≥ min_raw` | 东西**真在**画面里没 | 只看 lift：基线低的 query 被任何画面变动带着一起触发 |

---

## 3. 启动探测：负样本从哪来、阈值怎么定

```mermaid
flowchart TB
    S["开机"] --> COLLECT["收集头 4 秒的帧嵌入<br/>没有 query 也照收"]
    COLLECT --> SCENE["scene = 均值并归一化<br/>= 这个环境长什么样"]

    SCENE --> MODE{"negatives.mode"}
    MODE -->|"off"| NONE["不用负样本<br/>纯余弦，零配置"]
    MODE -->|"manual"| MAN["用 yaml 里手写的"]
    MODE -->|"auto"| RANK["174 条词表<br/>逐条与 scene 算相似度"]

    RANK --> EXCL["排除与现有 query<br/>相似度 > 0.90 的<br/>（免得把自己的 query 否掉）"]
    EXCL --> QUOTA["按类别限额挑 K 条<br/>场景/物体/人/画质 每类 ≤ 2"]
    QUOTA --> NEG["负样本定了"]

    NEG --> CAL["用同一批探测帧<br/>给每条 query 重新打分"]
    MAN --> CAL
    CAL --> STAT["μ, σ"]
    STAT --> MR["min_raw = clip(<br/>　max( μ+3σ,<br/>　　　min(chance×1.2, (μ+3σ)×headroom) ),<br/>　floor, cap)"]
    STAT --> DO["delta_on = max(4σ, 0.2×min_raw, floor)"]
    MR --> READY["逐条 query 各有各的阈值"]
    DO --> READY

    NEW["运行中新注册一条 query"] -.->|"复用存下来的探测帧"| CAL

    classDef auto fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef calc fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    class RANK,EXCL,QUOTA,NEG auto
    class STAT,MR,DO calc
```

**为什么阈值必须逐条标定**：不同 query 在同一场景下的分数量纲能差 4 倍以上；
全局一个阈值会让后注册的 query 永远够不到。

**为什么 chance 锚点要加天花板**：`1.2/(1+K)` 是固定值，而分数量纲随负样本贴合度
差一个数量级——负样本越像这个场景，query 抢到的 softmax 概率越低。厨房场景实测
所有 query 峰值只有 0.017–0.094，固定的 0.171 是谁都翻不过的墙。

---

## 4. 语音通路

```mermaid
flowchart TB
    M["麦克风 16 kHz"] --> CAL["开头 1.5 秒测环境底噪<br/>阈值 = 底噪 × 2"]
    CAL --> VAD

    VAD["RMS VAD<br/>连续 2 块够响才算起音<br/>静音 0.7 秒算说完"]
    VAD --> TRIM["剪掉前后静音<br/>（不剪的话均值被稀释，<br/>真人声会被判成「太轻」，<br/>Whisper 也更容易幻觉）"]
    TRIM --> CHK{"有效时长 ≥ 0.45s<br/>且 能量 ≥ 底噪×1.6 ?"}
    CHK -->|"否"| DROP["丢弃（多半是东西碰桌子那一声）"]
    CHK -->|"是"| ASR

    ASR["Whisper<br/>transcribe → 原文<br/>translate → 英文"]
    ASR --> HAL{"是静音幻觉？<br/>Thank you for watching / 单字符刷屏"}
    HAL -->|"是"| DROP
    HAL -->|"否"| CMD

    CMD{"管理指令？<br/>中英文各查一遍"}
    CMD -->|"清空"| CLR["移除全部 query 及其状态"]
    CMD -->|"列出"| LST["打印当前任务与阈值"]
    CMD -->|"否"| INT

    INT{"意图判别<br/>原文 + 英译，任一命中即算"}
    INT -->|"STANDING"| VP["剥壳成画面描述<br/>「拿起手机就提醒我」<br/>→ a hand holding a phone"]
    INT -->|"INSTANT"| EMIT["直接出 WhenEvent<br/>不经过视觉门"]
    VP --> REG["注册到视觉门<br/>立刻用探测帧标定阈值"]

    classDef audio fill:#e0f7fa,stroke:#00838f,color:#006064
    classDef drop fill:#fafafa,stroke:#bdbdbd,color:#757575
    class CAL,VAD,TRIM,ASR audio
    class DROP drop
```

**说中文、query 用英文**：SigLIP 的文本塔是英文训练的，中文描述匹配不上。
Whisper 自带 `translate` 任务，任何语言进、英文出。注意 `large-v3-turbo`
**不支持翻译**，喂它 `task="translate"` 会原样返回中文。

---

## WHEN → WHICH 的契约

`route=SILENT` 的事件也会输出——既是心跳，也保留连续分数供后续标定阈值。

```json
{
  "seq": 106,
  "t_emit": 11.501,
  "route": "TRIGGER",
  "trigger_type": "STANDING",
  "query": {"text": "a water bottle placed close to a keyboard",
            "origin": "user_standing", "query_id": "sq_bottle"},
  "score": 0.1029, "threshold": 0.1,
  "raw_score": 0.7715, "baseline": 0.6017,
  "evidence": {"window": [9.501, 11.501],
               "frame_idx": [19, 20, 21, 22, 23],
               "novelty": 0.1698},
  "urgency": "normal",
  "cooldown_until": 16.501
}
```

| 字段 | WHICH 拿它做什么 |
|---|---|
| `trigger_type` | INSTANT 走正常模型选择；ALERT 延迟预算更紧 |
| `query.text` | 决策函数的 query 嵌入输入 |
| `evidence.frame_idx` | **HOW MUCH 的直接输入**——送几帧下去 |
| `score` / `raw_score` / `baseline` | 门有多确信，可作为升级信号 |
| `urgency` | 延迟预算 |
| `novelty` | 画面变化量，可作为额外特征 |

---

## 现状

| | 状态 |
|---|---|
| WHEN 视觉门 | 已实现，零训练 |
| WHEN 语音通路 | 已实现 |
| WHEN 探测与标定 | 已实现 |
| **WHICH 全部** | **未实现**——上图中的设计来自架构讨论，尚未落地 |
| 记忆系统接口 | 未实现，`WhenEvent` 里已预留位置 |

WHEN 的已知弱项见 [CHANGELOG.md](CHANGELOG.md) 的 TODO：跨场景的分数量纲差一个
数量级，阈值公式目前靠三个约束拼出来，`auto` 仍略逊于 `manual`。
