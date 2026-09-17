# Memory Eval Harness 数据契约

状态：第一版类型草案，尚无实现。本文补充[主设计](eval-harness.md)中的接口，保留已确认的评测协议；字段名称与序列化规则是供实现、review 使用的建议，不表示已经逐字段确认。已按 Evidence 结构评审修订：证据类型不再携带记忆身份与有效性，操作目标统一取自回执、状态断言统一走 `inspect`，并补充来源时间渲染与证据渲染开销的约束。整体设计评审带来的接口变化（指标注册表、逐题归因分类、预算语义、显式更新条件）同样记录在文末[修订记录](#修订记录)。

Memory adapter 接入的是整个被测记忆系统。`Evidence` 表示它检索返回的内容，不表示 episodic、semantic 或 procedural 等内部记忆层。把接口返回值、Reader 实际输入及私有评分数据分开，使预算、召回、成本和恢复检查可以落到具体字段。

## 通用约定

- 以下 Python 类型用于表达结构；实施时以 pydantic v2 模型作为单一真源，文档中的 dataclass 视为等价语义草图，运行时校验、序列化与 `schema_version` 都由它承载。所有列出的字段都必须存在；`T | None` 表示值可为 JSON `null`，不是省略字段。列表无元素用 `[]`。
- 来源只使用清洗后的匿名内部 ID。`namespace`、`sample_handle`、`operation_id` 不编码上游样本 ID、gold 或拒答类别。稳定 ID 的作用域写入配置；同一空间重新打开及操作重试后保持不变。
- 时间为数据集转换后的日期或时间字符串，保留实际精度和已知时区；日期可以是 `2026-09-06`。不能给只有日期的数据补造时区，也不能用机器当前时间代替 `question_date`。运行尝试的起止时间另外使用 UTC 时间戳。
- 空集合表示当前输出确实没有条目；未知标量用 `null`。失败记录在调用尝试或阶段中，不用空证据冒充错误。成本和调用次数只聚合调用尝试中的用量。

## Memory adapter 可见的数据

```python
from dataclasses import dataclass
from typing import Literal

Validity = Literal["current", "superseded", "deleted", "unknown"]

@dataclass(frozen=True)
class SourceRef:
    session_id: str
    msg_id: str

@dataclass(frozen=True)
class SourceSpan:
    session_id: str
    msg_id: str
    start: int
    end: int

@dataclass(frozen=True)
class Message:
    msg_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str

@dataclass(frozen=True)
class Session:
    session_id: str
    occurred_at: str
    messages: list[Message]

@dataclass(frozen=True)
class QueryContext:
    query: str
    question_date: str

@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    question_date: str
    evidence_token_budget: int

@dataclass(frozen=True)
class Evidence:
    # 不含记忆身份与有效性：检索单元的身份由 harness 的 raw_index 分配，
    # 操作目标取自回执，状态只能按稳定 ID 通过 inspect 查询。
    kind: Literal["extractive", "generated"]
    text: str
    extractive_span: SourceSpan | None
    derivation_sources: list[SourceRef]
    source_times: list[str]
    retrieval_score: float | None

@dataclass(frozen=True)
class MemoryState:
    memory_id: str
    content: str | None
    sources: list[SourceRef]
    validity: Validity
    superseded_by: list[str] | None

@dataclass(frozen=True)
class ResourceUsage:
    input_tokens: int | None
    output_tokens: int | None
    llm_call_count: int | None
    cost_amount: str | None
    currency: str | None

@dataclass(frozen=True)
class ErrorInfo:
    code: str
    message: str
    effect: Literal["none", "possible", "confirmed"]
    transient: bool

@dataclass(frozen=True)
class MutationReceipt:
    operation_id: str
    status: Literal["accepted", "completed", "failed"]
    memory_ids: list[str]
    sources: list[SourceRef]
    error: ErrorInfo | None
    usage: ResourceUsage | None
```

### 输入与证据约束

`Session` 只包含当前会话。消息采用白名单字段，不能保留 `has_answer`、工具调用附加对象或上游标注。首版只转发统一清洗得到的文本；`role` 集合是允许值，不要求 adapter 生成缺失的角色。会话时间不强加到没有独立时间的每条消息上。

Dataset 向 Runner 提供 QueryContext，Runner 将相同 query/question_date 交给 Reader，并附加固定预算构造 RetrievalRequest 交给 Memory；Reader 不需要通过检索请求获得查询上下文。`RetrievalRequest` 的预算是正整数，namespace 单独传入。`retrieve(namespace, request) -> list[Evidence]` 保持该签名，按 adapter 排名顺序返回；成功返回 `[]` 表示没有证据。超时、后端异常或非法输出应产生错误尝试，不能包装成成功的 `[]`。

`Evidence.text` 必须非空。`retrieval_score` 为有限数字或 `null`，不跨实现比较；它只作诊断，harness 不要求它与返回顺序单调一致，排序一律以 adapter 的返回顺序为准。

Evidence 不携带记忆有效性。有效性回答的是“某个稳定目标当前处于什么状态”，只有在能按 ID 指定目标时才有意义；而 `retrieve` 返回的是受排名和预算影响的 top-k，无法按请求的 ID 取回指定目标，因此不能承担状态查询职责。“被替代”“已删除”这类状态一律通过 `inspect` 读取 `MemoryState`；缺少该能力时，依赖状态的操作为 `not_supported`，不能改用 Evidence 的字段代替。

Evidence 也不携带记忆身份。它引用的来源（`session_id`/`msg_id`）是可核验的，而“这条返回内容属于哪一条可操作记忆”属于被测实现的内部组织，harness 既不校验也不依赖：检索单元的身份由 `raw_index` 在准备阶段分配，显式 update/delete 的目标 ID 一律取自 `MutationReceipt`，状态通过 `inspect` 按稳定 ID 查询。需要稳定 ID 的能力在 `capabilities()` 中声明，而不是靠证据上的可选字段表达。

| 输出形态 | 必须满足的约束 | 评分用途 |
| --- | --- | --- |
| `extractive` | `extractive_span` 非空；`derivation_sources=[]`；`text == 清洗消息.content[start:end]` | 通过核验且在预算内的实际范围可以计原文 recall |
| `generated` | `extractive_span=null`；生成文本不伪装成某段原文；已知来源写入 `derivation_sources` | 文本可供 Reader，来源只用于诊断追踪，不计原文 recall，也不作为删除或更新是否生效的依据 |

`start/end` 是 Python Unicode 字符索引，采用半开区间 `[start, end)`，满足 `0 <= start < end <= len(content)`；不是字节或 token 偏移。索引基于最终清洗消息，不能再隐式规范化文本。一个原文单元只能引用一个消息范围；跨消息拼接应拆成多个 Evidence。

`source_times` 表示内容来源的已知时间，按出现顺序去重，使用约定格式的日期或带时区的时间字符串。原文单元至少含其会话时间并与清洗记录一致；生成内容可以含多个来源时间，未知时为 `[]`，不以摘要创建时间充当来源时间。Reader 看到的时间由 harness 按固定格式渲染，adapter 提供的字符串不直接作为自由文本进入上下文；无法按约定格式解析的时间不渲染到 Reader，只保留在诊断记录中并在结果里标记。展示的已知来源时间所占 tokens 计入预算。generated 的 `derivation_sources=[]` 表示不能提供已知来源，诊断中明确标为来源未知，不伪造来源。`derivation_sources` 只保存在诊断结构中，不渲染给 Reader，也不作为删除或更新是否生效的判定依据；可附带的原文必须作为另外的 extractive Evidence 返回并占用预算。

Evidence 可以返回过程或方法文本，这只规定传输格式；流程提炼、适用性选择及执行收益的 procedural suite 仍需另外设计。

### 两个 Evidence 示例

假设清洗会话 `s_6f2c` 的消息 `m_91ab` 内容恰好是“迁移到 pnpm，以后都用 pnpm。”，日期为 `2026-09-03`。该消息共 19 个 Unicode 字符：

```json
{
  "kind": "extractive",
  "text": "迁移到 pnpm，以后都用 pnpm。",
  "extractive_span": {"session_id": "s_6f2c", "msg_id": "m_91ab", "start": 0, "end": 19},
  "derivation_sources": [],
  "source_times": ["2026-09-03"],
  "retrieval_score": 2.73
}
```

同一历史生成的摘要可以是以下形式。示例中的来源声明供追踪，不足以证明摘要正确，也不能代替上例的原文证据：

```json
{
  "kind": "generated",
  "text": "项目目前统一使用 pnpm。",
  "extractive_span": null,
  "derivation_sources": [{"session_id": "s_6f2c", "msg_id": "m_91ab"}],
  "source_times": ["2026-09-03"],
  "retrieval_score": null
}
```

两个示例都不含记忆身份与状态字段：证据的身份由 `raw_index` 分配，记忆状态由 `inspect` 返回的 `MemoryState` 表达。

### 状态、回执与用量

`inspect` 对每个请求的稳定 ID 返回一个 MemoryState。支持状态检查的实现应保留可观察的删除状态，不能用遗漏条目让 runner 猜测已经删除。找不到 ID 时仍返回该 ID，`validity=unknown`、`content=null`，这不证明删除成功；已确认删除返回 `deleted` 状态或等价 tombstone。`deleted/unknown` 的内容允许为 `null`；`current/superseded` 必须有内容。`superseded_by=[]` 表示已知没有替代项，`null` 表示后端没有提供替代关系。`sources=[]` 表示没有可报告的来源，不能据此通过依赖来源关系的断言。

普通问答可以不支持 `inspect`；该能力未声明时，对应操作为 `not_supported`。已经声明状态能力却返回 `unknown`、遗漏状态或无法提供必要关系时，对应断言失败，不能改为不支持。

`accepted/completed` 回执的 `error=null`；`failed` 必须带错误。`memory_ids=[]` 允许出现在尚未分配 ID、没有生成可操作条目或失败的回执中；但依赖稳定 ID 的显式更新/删除测试必须获得目标 ID，否则不能通过。`sources` 只列此次操作关联的内部来源。

`completed` 的完成条件见主设计：持久化、索引可见且重新打开后可观察。`accepted` 必须继续等待同一 `operation_id`；`failed` 不能代替完成。回执的失败不保证修改未生效：`error.effect=possible` 表示结果不确定，必须按幂等能力查询原操作或隔离空间重放；仅 `effect=none` 且确认不会稍后生效时才允许按明确未生效规则重试。`transient=true` 本身不授权重试修改。

同一 operation ID 重试必须对应完全相同输入；同 ID 不同输入拒绝。`await_ready` 不能再次提交修改，`close` 不能代替等待。幂等和状态查询能力在运行前声明并固定。

ResourceUsage 的所有未知数量为 `null`，已知没有调用为零；token 和调用次数为非负整数。费用用十进制字符串与币种表示，未知时二者均为 `null`，不同币种不能直接求和。没有返回任何用量时 `usage=null`，不等同于全零。

用量属于一次调用尝试，不属于每条 Evidence。`MutationReceipt.usage` 若提供，是此次提交或状态查询尝试的用量，不是累计任务总量；同一数据记录到下文 StageAttempt 时只聚合后者。等待接口不能重复返回已计过的构建用量。Memory 的检索等无回执接口通过 runner 提供的 `UsageRecorder`（基于 contextvar）上报用量：调用前 runner 建立记录器，调用结束后把结果写入对应的 StageAttempt；未上报即为未知，不用零填充。提供累计计量的后端需转换为可去重的尝试记录，无法确定的部分标为未知。

### 能力声明词汇

`capabilities() -> set[str]` 在运行前返回首版已知能力词汇。写入、只读检索、独立空间及完成确认是接入的基础契约，不因为缺少可选能力而取消；未知字符串可留作扩展诊断，但不能自动使测试通过。

| 字符串 | 具体承诺 |
| --- | --- |
| `extractive_evidence` | 提供可验证的原文证据；适用题未返回原文时 recall 为零 |
| `generated_evidence` | 可以返回生成内容；声明本能力不代表内容或来源已经验证 |
| `state_inspection` | 通过 `inspect` 按稳定 ID 返回指定目标的 `MemoryState`；检索结果不能代替，不依靠自由文本猜测 |
| `auto_update` | 连续 ingest 能使旧、新约定状态变化，供自动更新测试检查 |
| `update` | 支持按稳定 memory_id 替换完整文本；完成后 `inspect` 必须给出 `content=replacement`、`validity=current`，旧文本不得再作为当前值返回 |
| `delete` | 支持按稳定 memory_id 删除，目标及受影响衍生内容不再可召回 |
| `async_mutation` | 修改可能返回 accepted，提供 await_ready 直到明确终态 |
| `idempotent_mutation` | 相同 operation_id、相同输入仅产生一次逻辑修改；不同输入拒绝 |
| `operation_status` | 不重新提交修改即可查询/等待同一 operation_id 的状态 |

证据模式由前两个能力及基线类型共同固定：纯生成不参加原文 recall；混合输出只对原文部分计该指标。真实检索 adapter 至少声明一种证据输出能力，无记忆基线按其 reader-only 协议例外处理；完整历史基线仍不参加排名 recall。

async_mutation 必须同时提供 operation_status 和 await_ready；operation_status 不隐含幂等，幂等也不隐含可查询状态。显式 update/delete 需要稳定目标 ID。auto_update、update 测试的状态断言需要 state_inspection，且必须通过 `inspect` 按稳定 ID 查询：检索返回受排名和预算限制、不能指定目标，因此不构成等价的状态暴露方式；缺少该必要能力在运行前标为不支持，声明支持后却返回 unknown 则失败。删除测试同时检查目标、衍生内容及无关记忆，以这些记忆是否仍可召回为准，不用空检索结果或 adapter 自报的衍生来源单独推断操作通过。

## Harness 内部的数据

以下类型接在前一段 Python 定义后，不传给 Memory。Reader 只接收查询上下文和 PreparedEvidence 的 `rendered_text`，不接收整个内部对象。

```python
@dataclass(frozen=True)
class PreparedItem:
    raw_index: int
    evidence: Evidence
    verified_span: SourceSpan | None
    rendered_text: str
    token_count: int
    retained_chars: int
    truncated: bool

@dataclass(frozen=True)
class PreparedEvidence:
    rendered_text: str
    items: list[PreparedItem]
    token_count: int
    text_token_count: int
    budget: int | None
    counting_mode: Literal["exact", "estimated", "test"]
    tokenizer_id: str
    dropped_raw_indices: list[int]

@dataclass(frozen=True)
class ReaderResult:
    hypothesis: str
    raw_output: str
    model: str | None
    usage: ResourceUsage | None

@dataclass(frozen=True)
class ScoringData:
    expected_answer: str
    gold_source_ids: list[str]
    internal_to_official_session: dict[str, str]
    is_abstention: bool
    question_type: str
    official_fields: dict[str, object]

@dataclass(frozen=True)
class JudgeRequest:
    question: str
    expected_answer: str
    hypothesis: str
    question_type: str
    protocol_id: str
    protocol_fields: dict[str, object]

@dataclass(frozen=True)
class JudgeResult:
    correct: bool
    raw_output: str
    model: str | None
    usage: ResourceUsage | None

@dataclass(frozen=True)
class StageAttempt:
    attempt_id: str
    stage: str
    operation_id: str | None
    outcome: Literal["running", "returned", "error"]
    started_at: str
    ended_at: str | None
    elapsed_ms: float | None
    input_ref: str
    output_ref: str | None
    error: ErrorInfo | None
    usage: ResourceUsage | None

@dataclass(frozen=True)
class MetricResult:
    metric_id: str  # 必须来自指标注册表，未登记的指标只能作为明确标记的诊断项
    status: Literal["computed", "not_supported", "not_applicable", "pending"]
    value: float | None
    reason: str | None

Attribution = Literal["hit_correct", "hit_wrong", "miss_correct", "miss_wrong"]

@dataclass(frozen=True)
class Result:
    run_id: str
    sample_handle: str
    namespace: str
    config_fingerprint: str
    suite: Literal["qa", "operations"]
    qa_status: Literal["pending", "scored", "failed", "context_exceeded", "invalid_input"] | None
    operation_status: Literal["pending", "passed", "failed", "not_supported"] | None
    correct: bool | None
    attribution: Attribution | None
    failed_stage: str | None
    metrics: list[MetricResult]
    stage_states: dict[str, str]
    artifact_refs: dict[str, str]
    attempts: list[StageAttempt]
```

### Reader 输入准备与评分边界

Runner 给一次 retrieve 的返回列表分配从零开始的 `raw_index`，因此不要求 adapter 为检索返回单元生成稳定 Evidence ID。`PreparedItem.evidence` 是实际保留的内容：原文前缀裁剪时同时修改 `text` 和 `extractive_span.end`；生成内容裁剪仍保持 generated。`verified_span` 仅对已经通过核验的 extractive 单元非空，必须与实际范围相同。

`retained_chars` 为保留文本的 Unicode 字符数；`truncated` 表示相对该原始单元发生裁剪。完全移除的单元不在 items 中，而进入 `dropped_raw_indices`。每个非空原始单元恰好属于一个保留项或移除项。来源不存在、范围无效或内容不匹配属于非法 adapter 输出，使准备阶段失败；不能悄悄改成 generated 或拿清洗历史补齐内容。

`rendered_text` 是 Reader 真正收到的证据部分，不能渲染 derivation_sources 或 gold。PreparedEvidence 的 token_count 对完整渲染文本重新计数，包含来源元数据、分隔符与格式标记。`PreparedItem.token_count` 是该单元渲染片段（其文本加该条自身的来源元数据）的 tokens，仅作诊断；`PreparedEvidence.text_token_count` 是各单元保留文本分别计数后的总和，用于估算文本与格式开销的比例。由于 tokenizer 的边界效应，两者之差不保证恰好等于逐项元数据与分隔符的开销，因此只作诊断，不参与预算判定。单元粒度由 adapter 决定，碎片化返回会抬高每条证据的元数据开销，报告据此区分“粒度差异”和“检索质量差异”。预算为 4096 时必须满足总量不超预算；完整历史基线的 `budget=null` 表示采用其独立上下文预检规则。无记忆时 `items=[]`、文本为空、计数为零。`test` 计数模式仅用于 M1，不与真实模型结果比较。

ScoringData 只在 Scorer 的私有视图可见，内部 ID 映射及 official_fields 不送 Memory、Reader 或通用输入准备函数。Runner 可以保存私有追踪引用，Reporter 可以读取必要分类字段，但不能将这些数据作为被测组件的输入。非拒答样本的 gold_source_ids 必须非空且映射完整；拒答题的 recall 为 N/A，不从数组是否为空猜测拒答标志。

JudgeRequest 是通用语义输入；官方协议 adapter 按固定版本将其转换为实际 prompt，`protocol_fields` 只允许该协议所需的私有字段，不能自创 prompt 替代官方协议。judge 所用模型与官方验证过的模型不同时，仍沿用官方模板与解析语义，但必须在记录中标注该偏离，不得宣称与论文分数可比。JudgeResult.correct 由官方协议解析成功后生成；输出不可解析时评分阶段失败，不能默认判错。ReaderResult/JudgeResult 都只表示已成功产生并接受的输出，失败不伪造对象。两者的 `model` 记录服务端响应中报告的模型标识，未提供时为 `null`；它和该次运行的别名、厂商标注版本与运行日期一起用于判定模型是否漂移。用量对象是同一次 StageAttempt 的镜像，资源汇总不重复累计。

## 进度、结果与检查

StageAttempt 记录每次真实尝试；accepted 回执属于成功返回的提交尝试，但该逻辑写入阶段仍未完成。stage_states 至少区分 `pending/completed/failed`，正式枚举及阶段名称在 M1 固定。running 尝试的结束、耗时和输出可以为 null；错误尝试必须有 error，成功返回必须有 output_ref，引用可以指向返回 None 的完成事件。

Result 仅承载最小进度及产物索引，详细输入、原始 Evidence、PreparedEvidence、回执、Reader/Judge 输出及 ScoringData 存在 artifact_refs 对应的产物中。引用必须能解析到本 run 的带校验值记录。断点恢复检查配置指纹和空间身份，不覆盖成功阶段的产物；重试/重放生成新 attempt_id，修改 operation_id 沿用原逻辑 ID。

`attribution` 只在 qa suite 且 `qa_status=scored` 时非空，它是主设计中联合归因分布的逐题输入。命中维度的口径按该题声明的证据模式判定：可核验原文检索取 session recall > 0，其余对照取是否提供了非空证据，无记忆基线恒为未命中。failed、context_exceeded、pending 与 invalid_input 不给归因分类，不并入“回答错误”。

qa suite 的 operation_status 为 null；operations suite 的 qa_status/correct 为 null。QA `scored` 时 correct 必须为布尔值；failed、context_exceeded、pending 时为 null，不捏造错误回答。失败按零分计入主设计的整体指标是 Reporter 的聚合规则，不通过设置 correct=false 混淆失败和已评分错误。invalid_input 阻塞或使正式 run 无效。

`metric_id` 必须来自指标注册表；注册表未登记的指标只能作为明确标记的诊断项，不进入比较和排名。MetricResult.computed 必须有数值，其他状态为 null 并附 reason。原文检索适用题发生写入/检索失败时，recall 仍按空 R 计算为零，并保留失败阶段；生成模式、拒答题或无排名基线按各自原因记 N/A。Reader/Judge 失败保留已经完成的 recall，后续恢复复用实际证据。操作失败与不支持分别记录，不用 unknown 算通过。

实施时至少检查：类型及必填字段、时间精度、匿名 ID 隔离、原文范围与文本一致、generated 来源不参与原文 recall、不渲染诊断来源、裁剪后证据一致、完整 token 预算、回执完成及错误效果、稳定操作 ID、阶段输出存在性、费用去重及恢复指纹。Evidence 不含记忆身份与状态字段，操作目标只取自 `MutationReceipt`、状态断言只经 `inspect` 取得；无法按约定格式解析的来源时间不进入 Reader；文本与格式开销可分别统计。两个 JSON 示例应能按 Evidence 契约加载；带错偏移、伪造来源、未知失败效果及在证据上携带身份或状态的样本必须被检测。

尚待 M1 落实的内容为序列化 schema 版本、实际 Python 校验库、产物引用格式及 Memory 调用用量的传递机制；这些选择不改变本文的数据边界。真实 tokenizer、模型及官方 judge 协议版本仍依主设计在 M2 前固定。

## 修订记录

本节记录历次评审后的修订。除标注“待确认”的条目外，以下选择作为当前设计基准；替代方案保留，供后续需求变化时重新评估。

| 修订 | 理由 | 替代方案 | 状态 |
| --- | --- | --- | --- |
| 从 Evidence 移除 `validity` | 状态属于稳定目标而不是证据本身；`deleted` 与“删除后不再被召回”矛盾；`retrieve` 无法按指定 ID 取回目标，Evidence 也没有 `content`、`superseded_by`，无法等价于 `inspect` | 保留一个只表示陈旧度的 `current/superseded/unknown` 字段，并分别定义其渲染与断言用途；若出现只能在检索路径暴露状态、且必须以证据标注满足状态断言的后端，再评估是否引入 | 已采纳 |
| 从 Evidence 移除 `memory_id` | 与 `validity` 同一病因：没有任何指标或断言消费该字段，固定返回 `null` 也不影响结果；update/delete 的目标 ID 实际取自 `MutationReceipt`，状态来自 `inspect`。保留它还会诱发“证据身份就是操作目标”的误解 | 保留为诊断元数据，只供人工排查；或要求声明 `update`/`delete` 的后端在相关证据上提供 ID 并纳入断言。前者仍是无校验字段，后者增加 adapter 负担而没有新指标 | 已采纳 |
| `derivation_sources` 只作诊断 | 来源由 adapter 自报，不能用来证明衍生内容已删除或已更新；原措辞“用于更新/删除检查”超出实际能力 | 将“未提供来源”一律判失败；与“来源未知不伪造来源”的约定冲突，暂不采用 | 已采纳 |
| 新增 `PreparedEvidence.text_token_count` | 单元粒度影响元数据开销，原结构无法把证据文本与格式开销分开报告 | 只保留逐项诊断计数，不新增字段；代价是报告无法区分粒度差异与检索质量差异 | 已采纳 |
| 来源时间由 harness 固定格式渲染 | 不可核验的自报时间不应成为进入 Reader 的自由文本通道 | 原样透传 adapter 字符串；与整体防伪造基调不一致 | 已采纳 |
| `retrieval_score` 明确为纯诊断 | 排序以返回顺序为准，分数不跨实现比较 | 校验同一次返回内分数与顺序单调一致；会额外约束不承诺排序质量的后端，暂不采用 | 已采纳 |
| 引入指标注册表，`MetricResult.name` 改为 `metric_id` | 指标名原为自由字符串，比较命令要求“指标定义一致”却无从检查，两个 run 可能在不同 k 或不同口径下被当成同条件比较 | 只在文档里约定命名；不可检查，跨 run 仍可能错配 | 已采纳 |
| 新增逐题归因分类 `Result.attribution` | 设计目标要求区分“未找到证据”与“找到证据但回答错误”，原指标各自独立汇总，无法直接量化 | 由 Reporter 在聚合时临时计算，不落入逐题记录；会失去按题追溯和按下钻分类的能力 | 已采纳 |
| 明确预算语义：harness 硬上限、adapter 提示性义务 | 原设计未说明 adapter 是否必须自行收敛，打包策略差异会无法解释地进入结果 | 把预算设为 adapter 的硬义务、越界即判失败；对第三方 adapter 不友好，且 harness 仍需兜底截断 | 已采纳 |
| 补齐显式更新的完成条件 | 原设计只要求“目标及衍生内容被正确更新”，未定义 inspect 可观察状态与来源要求，断言不可比 | 交由各 adapter 自定义；跨实现结果不可比 | 已采纳 |
| 组件表与工程结构补 `judges/`，`evidence/` 改名 `prepare/` | 官方协议 adapter 只出现在数据契约中，组件表与目录都没有对应位置，命名也不一致 | 保留原目录名并在文中说明映射；读者仍需自行对应 | 已采纳 |
| Reader 与 Judge 使用不同家族模型（Judge 定为 GLM-5.3） | reader 与 judge 同模型时，自偏好偏差随各实现产生的答案文本变化，共享盲区还可能奖励检索更差的实现；统一模型只能消除跨条件不可比 | 同模型加事后人工校准；偏差仍不可控，且无法解释 2×2 归因 | 已采纳 |
| `ReaderResult` / `JudgeResult` 增加 `model` 字段 | 两个模型都是滚动别名，需要按次记录服务端实际返回的模型标识，作为漂移检测与历史审计的依据 | 只依赖运行级配置快照记录版本；无法发现运行期间或跨运行的实际模型替换 | 已采纳 |
| 数据契约以 pydantic v2 模型作为单一真源 | dataclass 不做运行时校验；文档一套、校验一套会随时间漂移 | 保留文档 dataclass，另写校验模型 | 已采纳 |
| 用量通过 contextvar 的 `UsageRecorder` 上报 | 原设计把传递机制留给实施，而 `retrieve` 签名固定、无法直接返回用量 | 修改 `retrieve` 等签名返回 `(结果, 用量)`；会破坏已定义的接口 | 已采纳 |

以上修订不改变 M1 仍需落实的事项：产物引用格式与迁移兼容行为在 M1 前固定。数据契约以 pydantic v2 模型为单一真源，`schema_version` 首版从 1 起；指标注册表是整体设计评审新增的 M1 交付物，其版本随配置指纹保存。模型与数据选型的确定项记录在主设计的“已确定事项与仍待定事项”一节。
