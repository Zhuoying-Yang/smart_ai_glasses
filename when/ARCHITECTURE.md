# Smart AI Glasses Routing Architecture

基于 `zhuoyingyang-routing-full-pipeline@196db67`。

绿色为已接通，黄色为已有代码但未接入在线链路，灰色虚线为接口占位。

## 1. Full pipeline

```mermaid
flowchart TB
    subgraph INPUT[Input]
        ARIA[Project Aria<br/>RGB 1408×1408 @ ~10 FPS]
        CAM[Mac camera]
        MIC[Microphone<br/>16 kHz mono]
        BRIDGE[Aria bridge<br/>rotate + resize + MJPEG]
        ARIA --> BRIDGE
    end

    subgraph WHEN[WHEN · When to invoke]
        SLOT[Latest-frame slot<br/>capacity = 1]
        GATE[VisualGate<br/>SigLIP @ 2 FPS]
        AUDIO[VAD → Whisper<br/>intent rules]
        EVENT[WhenEvent]

        CAM --> SLOT
        BRIDGE --> SLOT
        SLOT --> GATE
        MIC --> AUDIO
        GATE -->|STANDING / ALERT| EVENT
        AUDIO -->|INSTANT| EVENT
        AUDIO -->|register STANDING| GATE
    end

    subgraph INTEGRATION[Integration]
        QUEUE[Routing queue<br/>capacity = 4]
        ADAPTER[WhenWhichAdapter]
        QUEUE --> ADAPTER
    end

    subgraph WHICH[WHICH · Which model]
        FEASIBILITY[Network / backend / budget / RTT]
        CLASSIFIER[WearVQA task classifier]
        RISK[Empirical SMALL failure risk]
        DECISION[WhichDecision]

        ADAPTER --> FEASIBILITY --> CLASSIFIER --> RISK --> DECISION
    end

    subgraph EXECUTOR[Execution]
        SMALL[Galaxy Gemma E2B<br/>SMALL_1F]
        LARGE[LARGE backend<br/>not connected]
        SPEECH[Mac speech output]

        DECISION -->|SMALL_1F| SMALL --> SPEECH
        DECISION -->|LARGE_1F| LARGE
    end

    EVENT -->|TRIGGER + one RGB frame| QUEUE

    classDef live fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef partial fill:#fff8e1,stroke:#f9a825,color:#6d4c00
    classDef missing fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray:5 4,color:#616161
    class ARIA,CAM,MIC,BRIDGE,SLOT,GATE,AUDIO,EVENT,QUEUE,ADAPTER,FEASIBILITY,CLASSIFIER,RISK,DECISION,SMALL,SPEECH live
    class LARGE missing
```

## 2. 运行时线程与数据流

```mermaid
flowchart TB
    subgraph CAPTURE[主线程：持续取帧]
        READ[OpenCV 读取摄像头]
        RGB[画面转为 RGB]
        READ --> RGB
    end

    subgraph VISION[视觉线程]
        VSLOT[单帧缓冲<br/>容量 1，新帧覆盖旧帧]
        SAMPLE[每秒取 2 帧]
        STEP[运行视觉门]
        VSLOT --> SAMPLE --> STEP
    end

    subgraph VOICE[音频线程]
        CALLBACK[麦克风回调<br/>每块 50 ms]
        UQUEUE[语句队列<br/>不限长度]
        ASR[语音识别<br/>判断用户意图]
        CALLBACK --> UQUEUE --> ASR
    end

    subgraph ROUTING[路由线程：串行处理]
        RQUEUE[触发队列<br/>最多等待 4 条，满后丢弃新触发]
        ROUTE[WHICH 选择模型]
        EXEC[执行模型]
        RQUEUE --> ROUTE --> EXEC
    end

    RGB -->|覆盖写入| VSLOT
    RGB -->|保存副本| LATEST[即时提问使用的最新画面]
    STEP -->|视觉触发 + 当前画面| RQUEUE
    ASR -->|即时提问 + 最新画面| RQUEUE
    LATEST --> ASR
```

## 3. WHEN：语音注册、视觉监测与触发

```mermaid
flowchart TB
    subgraph SPEECH[语音通路]
        MIC[麦克风 16 kHz] --> VAD[RMS VAD<br/>底噪标定 1.5 秒<br/>静音 0.7 秒收句]
        VAD --> VALID{有效语音 ≥ 0.45 秒?}
        VALID -->|否| DROP[丢弃]
        VALID -->|是| ASR[Whisper 识别<br/>非英语翻译成英文]
        ASR --> CMD{管理指令?}
        CMD -->|清空 / 列出| MANAGE[管理监测任务]
        CMD -->|否| INTENT{用户意图}
        INTENT -->|INSTANT| DIRECT[直接生成 TRIGGER<br/>不经过视觉门]
        INTENT -->|STANDING| REWRITE[改写成视觉描述]
    end

    subgraph QUERIES[监测任务]
        PRESET[系统预置 ALERT]
        ACTIVE[STANDING + ALERT<br/>每条 query 独立保存状态]
        REWRITE --> ACTIVE
        PRESET --> ACTIVE
        MANAGE --> ACTIVE
    end

    subgraph VISUAL[视觉通路]
        FRAME[RGB 画面] --> ENCODE[SigLIP 编码<br/>每秒 2 帧]
        ENCODE --> NOVELTY[计算画面变化 novelty]
        NOVELTY --> REPROBE{auto 模式且<br/>novelty ≥ 0.35?}
        REPROBE -->|是| PROBE[重新观察环境 4 秒<br/>选择负样本并逐 query 标阈值]
        REPROBE -->|否| SCORE
        PROBE --> SCORE[逐 query 打分<br/>余弦或 softmax]
        ACTIVE --> SCORE
        SCORE --> EMA[快 EMA - 慢基线 = lift]
        EMA --> GATE{同时满足<br/>启动 ≥ 1.5 秒<br/>presence ≥ min_raw<br/>lift ≥ delta_on<br/>冷却结束?}
        GATE -->|否| SILENT[SILENT]
        GATE -->|是| TRIGGER[TRIGGER<br/>冷却 20 秒]
        TRIGGER --> LATCH[锁定到目标消失<br/>presence < 0.9 × min_raw]
    end

    FRAME -->|即时提问取最新画面| DIRECT
    DIRECT --> EVENT[WhenEvent]
    TRIGGER --> EVENT
    SILENT --> EVENT
    EVENT -->|仅 TRIGGER| PIPELINE[RoutingPipeline]

    classDef speech fill:#e0f7fa,stroke:#00838f,color:#006064
    classDef visual fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef event fill:#fff8e1,stroke:#f9a825,color:#6d4c00
    class MIC,VAD,VALID,ASR,CMD,INTENT,DIRECT,REWRITE,DROP,MANAGE speech
    class PRESET,ACTIVE,FRAME,ENCODE,NOVELTY,REPROBE,PROBE,SCORE,EMA,GATE,SILENT,TRIGGER,LATCH visual
    class EVENT,PIPELINE event
```

## 4. WHEN 如何把触发交给 WHICH

```mermaid
flowchart TB
    subgraph WHEN_OUT[1 · WHEN 输出]
        EVENT[WhenEvent]
        ROUTE[route<br/>TRIGGER / SILENT]
        QUERY[query.text<br/>问题或监测描述]
        DETAIL[其他触发信息<br/>类型 · 紧急程度 · 分数<br/>帧编号 · novelty · ROI]
        FRAME[触发时的 RGB 画面]

        EVENT --> ROUTE
        EVENT --> QUERY
        EVENT --> DETAIL
    end

    subgraph ADAPTER[2 · WhenWhichAdapter 转换]
        FILTER{route 是 TRIGGER?}
        DROP[丢弃 SILENT]
        MAP[query.text → question]
        CONSTANTS[暂时写死为 true<br/>网络可用 · LARGE 可用 · API 预算充足]

        FILTER -->|否| DROP
        FILTER -->|是| MAP
    end

    subgraph WHICH_IN[3 · WHICH 实际收到]
        SIGNALS[WhichSignals]
        QUESTION[question]
        AVAILABLE[3 个可用性标记]
        ROUTER[WhichRouter<br/>选择 SMALL 或 LARGE]

        SIGNALS --> QUESTION
        SIGNALS --> AVAILABLE
        QUESTION --> ROUTER
        AVAILABLE --> ROUTER
    end

    ROUTE --> FILTER
    QUERY --> MAP
    MAP --> SIGNALS
    CONSTANTS --> SIGNALS

    DETAIL -. 当前未传给 WHICH .-> UNUSED[暂未使用]

    FRAME -->|绕过 Adapter，不进入 WhichSignals| EXECUTOR[模型执行器]
    ROUTER -->|WhichDecision| EXECUTOR

    classDef live fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef missing fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray:5 4,color:#616161
    class EVENT,ROUTE,QUERY,FRAME,FILTER,MAP,CONSTANTS,SIGNALS,QUESTION,AVAILABLE,ROUTER,EXECUTOR live
    class DETAIL,UNUSED missing
```

## 5. WHICH policy and action space

```mermaid
flowchart TB
    INPUT[WhichSignals] --> FORCE_SMALL{force_small?}
    FORCE_SMALL -->|yes| SMALL[SMALL_1F]
    FORCE_SMALL -->|no| FORCE_LARGE{force_large and<br/>LARGE feasible?}
    FORCE_LARGE -->|yes| LARGE[LARGE_1F]
    FORCE_LARGE -->|no| NETWORK{network available?}

    NETWORK -->|no| SMALL
    NETWORK -->|yes| BACKEND{LARGE available?}
    BACKEND -->|no| SMALL
    BACKEND -->|yes| BUDGET{API budget OK?}
    BUDGET -->|no| SMALL
    BUDGET -->|yes| RTT{RTT ≤ 800 ms?}
    RTT -->|no| SMALL

    RTT -->|yes| TASK[TF-IDF + logistic regression<br/>predict one of 10 task types]
    TASK --> PRIOR[Lookup Laplace-smoothed<br/>SMALL failure risk]
    PRIOR --> LEARNED{learned_score supplied?}
    LEARNED -->|yes| LSCORE[score = learned_score]
    LEARNED -->|no| RSCORE[score = task risk]
    LSCORE --> PRESSURE[phone busy +0.10<br/>memory pressure +0.15]
    RSCORE --> PRESSURE
    PRESSURE --> THRESHOLD{score ≥ 0.40?}
    THRESHOLD -->|yes| LARGE
    THRESHOLD -->|no| SMALL

    SMALL --> SMALL_EXEC[Galaxy Gemma E2B]
    LARGE --> LARGE_EXEC[LARGE executor not connected]

    ACTIONS[Action enum<br/>SKIP · MEMORY_ONLY<br/>SMALL_1F · SMALL_MULTI<br/>LARGE_1F · LARGE_MULTI]
    ACTIONS -. current router returns only .-> SMALL
    ACTIONS -. current router returns only .-> LARGE

    GEMINI[Gemini client]
    OPENROUTER[OpenRouter client]
    GEMINI -. not wired .-> LARGE_EXEC
    OPENROUTER -. not wired .-> LARGE_EXEC

    classDef live fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef partial fill:#fff8e1,stroke:#f9a825,color:#6d4c00
    classDef missing fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray:5 4,color:#616161
    class INPUT,FORCE_SMALL,FORCE_LARGE,NETWORK,BACKEND,BUDGET,RTT,TASK,PRIOR,LEARNED,LSCORE,RSCORE,PRESSURE,THRESHOLD,SMALL,SMALL_EXEC live
    class GEMINI,OPENROUTER partial
    class ACTIONS,LARGE,LARGE_EXEC missing
```
