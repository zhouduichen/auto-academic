# L0 Optimizer-State Literature Search Implementation Plan

> **Execution mode:** 在当前任务中逐项执行并验收；不使用 Windows，也不创建子任务。步骤使用 checkbox (`- [ ]`) 跟踪。

**Goal:** 在 Mac 本地完成候选无关的 optimizer-state 定向文献检索，建立可追溯查询日志、主来源验证记录、直接威胁深读综述和可机械处理的 claim matrix。

**Architecture:** 先封存已有 40 篇候选与 12 篇深读记录的真实验证状态，再对六个预注册主题执行机制/数学/应用三类查询。搜索结果按 DOI、arXiv ID 和标题去重，仅用官方论文页、PDF、元数据和作者代码确定技术事实。所有可能等价实现持久优化器状态控制的工作都进入深读，最后通过两轮饱和检索关闭 L0 发现阶段。

**Tech Stack:** Mac shell，本地 ARIS 查询 helper，Web/arXiv/Semantic Scholar/OpenAlex，Markdown，CSV，JSONL，Git。

## Global Constraints

- 所有检索、验证、文档和轻量脚本工作只在 Mac 执行；本计划不连接、不更新、不提交 Windows。
- ARIS 只辅助发现、去重和留痕；论文事实只来自主来源。
- 不把 40 篇 ARIS 候选称为已验证；其当前状态是 `WARN/verify_papers_invocation_failed`。
- L0 保持候选无关；不在精确 recurrence 冻结前填写 candidate-overlap 结论。
- 预印本必须标注 `preprint`；正式发表版优先作为元数据主记录。
- 默认检索 2022–2026；只对必需基础工作向前回溯。
- 搜索摘要、AI 综述、二手博客和聚合页不支持 claim-level 事实。
- 不修改现有 `docs/research-direction/09_adjacent_work_claim_matrix.csv`；新矩阵单独建立并保留可追溯来源。

---

### Task 1: 封存起点与查询注册表

**Files:**
- Create: `.aris/literature/l0_search_log.jsonl`
- Create: `docs/research-direction/11_optimizer_state_literature_update.md`
- Read: `.aris/verify-papers/candidate_papers.json`
- Read: `.aris/verify-papers/verified_papers.json`
- Read: `docs/research-direction/09_adjacent_work_claim_matrix.csv`

**Interfaces:**
- Consumes: 现有候选、ARIS verifier verdict、12 篇深读矩阵和设计文档的 L0 执行合同。
- Produces: 查询截止时间、六个主题、18 条首轮查询、初始证据状态和后续日志的 JSONL schema。

- [ ] **Step 1: 盘点本地论文库和现有候选**

Run:

```bash
rg --files papers literature 2>/dev/null | sort
jq '{verdict,reason_code,summary,paper_count:(.papers|length)}' .aris/verify-papers/verified_papers.json
python3 - <<'PY'
import csv
from pathlib import Path
path = Path("docs/research-direction/09_adjacent_work_claim_matrix.csv")
with path.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
print({"deep_read_rows": len(rows), "work_ids": [row["work_id"] for row in rows]})
PY
```

Expected: 本地 PDF 库可明确列出或确认不存在；ARIS verdict 为 `WARN`；深读矩阵为 12 行。

- [ ] **Step 2: 写入查询日志的 schema 说明和首轮查询条目**

JSONL 每行使用以下字段：

```json
{"schema_version":1,"query_id":"T1-M1","theme":"optimizer_state_contamination","query":"optimizer state contamination gradient noise momentum Adam","source":"web","searched_at":"2026-07-31T00:00:00+08:00","cutoff_date":"2026-07-31","raw_result_ref":"not_searched","dedup_keys":[],"verification_status":"search_registered","notes":"mechanism query"}
```

18 条首轮查询必须覆盖：

```text
T1 optimizer_state_contamination:
  optimizer state contamination gradient noise momentum Adam
  persistent optimizer memory corrupted gradients momentum
  outlier gradient effect Adam moments future updates
T2 selective_moment_update:
  selective momentum update gradient optimizer
  Adam skip moment update preserve parameter update
  gated exponential moving average optimizer gradient consistency
T3 robust_adam_momentum:
  robust Adam heavy tailed corrupted stochastic gradients momentum
  robust momentum outlier gradient optimizer
  spike aware Adam momentum reset optimizer
T4 gradient_agreement:
  gradient agreement temporal consistency optimizer update
  gradient cosine similarity noisy labels optimizer
  gradient alignment sample selection persistent momentum
T5 noisy_peft:
  noisy label parameter efficient fine tuning LoRA optimizer
  robust PEFT noisy labels optimizer state
  LoRA label noise gradient momentum
T6 low_overhead_direct:
  single model single stage noisy label learning optimizer
  online noisy label robust training no clean set one forward pass
  lightweight noisy label training optimizer state 2026
```

- [ ] **Step 3: 建立文献更新文档骨架**

文档必须立即写入已知事实，不使用占位符：

```markdown
# 11 — Optimizer-State 定向文献更新

## 范围与截止时间
## 已有证据状态
## 检索主题与查询注册表
## 候选发现与去重
## 直接威胁深读
## 主题综合
## 饱和检查
## L1 待候选冻结的精确问题
```

- [ ] **Step 4: 验证 JSONL 可解析且查询 ID 唯一**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path(".aris/literature/l0_search_log.jsonl").read_text().splitlines() if line.strip()]
ids = [row["query_id"] for row in rows]
assert len(rows) == 18
assert len(ids) == len(set(ids))
assert {row["theme"] for row in rows} == {
    "optimizer_state_contamination", "selective_moment_update",
    "robust_adam_momentum", "gradient_agreement", "noisy_peft",
    "low_overhead_direct",
}
print("query registry valid", len(rows))
PY
```

Expected: `query registry valid 18`.

- [ ] **Step 5: 提交起点和查询注册表**

```bash
git add docs/research-direction/11_optimizer_state_literature_update.md
git commit -m "research: register L0 optimizer-state literature search"
```

`.aris/literature/l0_search_log.jsonl` 保持为本地 ARIS 痕迹，不加入当前 Git 提交。

### Task 2: 执行首轮多源发现与机械去重

**Files:**
- Modify: `.aris/literature/l0_search_log.jsonl`
- Modify: `docs/research-direction/11_optimizer_state_literature_update.md`
- Create: `docs/research-direction/12_optimizer_state_claim_matrix.csv`

**Interfaces:**
- Consumes: Task 1 的 18 条查询、现有 40 篇候选和 12 篇深读记录。
- Produces: 去重后的候选集、每个候选的来源/验证状态/纳入理由，以及直接威胁队列。

- [ ] **Step 1: 对 18 条查询执行 Web 与 arXiv 发现**

每条保存原始返回参考和搜索日期。技术结论不从搜索摘要中提取。

- [ ] **Step 2: 显式运行 Semantic Scholar 和 OpenAlex helper**

Run the locally resolved ARIS helpers with the six theme-level queries. Expected: each available helper returns structured JSON or a documented, non-silent failure. No Windows command is permitted.

- [ ] **Step 3: 按标准键去重**

Dedup precedence:

```text
DOI > arXiv ID > normalized(title)
```

Title normalization is Unicode NFKC, lowercase, punctuation removal, and whitespace collapse. Formal publication metadata replaces preprint venue metadata while retaining the arXiv link.

- [ ] **Step 4: 建立 claim matrix**

CSV columns are fixed as:

```text
work_id,title,year,status,venue,primary_url,doi,arxiv_id,code_url,
verification_status,theme,optimizer_family,state_target,current_update_effect,
persistent_state_effect,detector_signal,clean_reference_required,
noise_rate_required,extra_models,extra_forward_backward,extra_stage,
peft_evidence,noisy_label_evidence,mechanism_evidence,efficiency_evidence,
closest_overlap_class,direct_prior_risk,inclusion_decision,inclusion_reason,
source_queries,audit_notes
```

- [ ] **Step 5: 验证矩阵 schema 与去重约束**

Run:

```bash
python3 - <<'PY'
import csv
from pathlib import Path
path = Path("docs/research-direction/12_optimizer_state_claim_matrix.csv")
with path.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
assert rows
required = {"work_id", "title", "verification_status", "theme", "state_target", "inclusion_decision", "source_queries"}
assert required <= set(rows[0])
ids = [row["work_id"] for row in rows]
assert len(ids) == len(set(ids))
assert all(row["inclusion_decision"] in {"deep_read", "context", "exclude"} for row in rows)
print("claim matrix valid", len(rows))
PY
```

Expected: `claim matrix valid N`, with `N > 0`.

- [ ] **Step 6: 提交候选发现和矩阵**

```bash
git add docs/research-direction/11_optimizer_state_literature_update.md docs/research-direction/12_optimizer_state_claim_matrix.csv
git commit -m "research: map optimizer-state adjacent literature"
```

### Task 3: 主来源验证与直接威胁深读

**Files:**
- Modify: `.aris/literature/l0_search_log.jsonl`
- Modify: `docs/research-direction/11_optimizer_state_literature_update.md`
- Modify: `docs/research-direction/12_optimizer_state_claim_matrix.csv`

**Interfaces:**
- Consumes: Task 2 的 `deep_read` 队列。
- Produces: 公式级机制描述、当前/持久效应区分、设定与算力约束、直接先验风险。

- [ ] **Step 1: 核对每篇直接威胁的主来源**

Acceptable evidence is one or more of:

```text
official proceedings/journal page
official arXiv abstract/PDF
DOI metadata resolving to the publisher
author-linked official repository
```

Unavailable or ambiguous sources remain `unverified`; they are never silently removed.

- [ ] **Step 2: 深读方法与实验部分**

For each direct threat extract:

```text
exact state recurrence or update rule
whether the current batch's parameter effect is attenuated
which persistent tensors are gated/reset/robustified
signal used to detect an outlier or low-quality gradient
extra model/pass/stage/reference/noise-rate assumptions
PEFT and noisy-label evidence
wall-clock/VRAM/data-access accounting
published limitations and our independently observed boundary
```

- [ ] **Step 3: 分级直接先验风险**

Use only:

```text
critical: may implement the same persistent-state control contract
high: changes the same state carrier but not the current/persistent separation
medium: strong simple or direct baseline with a different mechanism
context: motivates the problem or evaluation protocol without mechanism overlap
```

- [ ] **Step 4: 将公式、设定和局限写入报告与矩阵**

Every technical statement in the narrative must carry a primary-source link or a matrix work ID whose `primary_url` is populated.

- [ ] **Step 5: 验证直接威胁无空证据字段**

Run:

```bash
python3 - <<'PY'
import csv
from pathlib import Path
with Path("docs/research-direction/12_optimizer_state_claim_matrix.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
direct = [row for row in rows if row["inclusion_decision"] == "deep_read"]
assert direct
for row in direct:
    for key in ("primary_url", "verification_status", "state_target", "current_update_effect", "persistent_state_effect", "direct_prior_risk", "audit_notes"):
        assert row[key].strip(), (row["work_id"], key)
print("direct threats complete", len(direct))
PY
```

Expected: `direct threats complete N`, with `N > 0`.

- [ ] **Step 6: 提交深读结果**

```bash
git add docs/research-direction/11_optimizer_state_literature_update.md docs/research-direction/12_optimizer_state_claim_matrix.csv
git commit -m "research: deep-read optimizer-state direct priors"
```

### Task 4: 饱和检索、综合与 L0 关闭

**Files:**
- Modify: `.aris/literature/l0_search_log.jsonl`
- Modify: `docs/research-direction/11_optimizer_state_literature_update.md`
- Modify: `docs/research-direction/12_optimizer_state_claim_matrix.csv`

**Interfaces:**
- Consumes: 已深读直接威胁的同义词、related-work 和引用链。
- Produces: 两轮无新 `critical/direct` 工作的饱和证据、候选无关的空白与风险结论、L1 精确问题清单。

- [ ] **Step 1: 根据直接威胁生成第一轮反查查询**

Include method aliases, recurrence terminology, titles/authors from related work, and forward/backward citation variants. Save every query as a new JSONL row with `query_id` prefix `S1-`.

- [ ] **Step 2: 执行第一轮反查并更新矩阵**

Any new direct threat returns to Task 3 before proceeding.

- [ ] **Step 3: 根据第一轮结果生成第二轮独立查询**

Second-round queries must not be verbatim repeats. Save them with `query_id` prefix `S2-`.

- [ ] **Step 4: 执行第二轮并判定饱和**

Saturation passes only if both S1 and S2 add zero new `critical` or `deep_read` entries after primary-source verification. Otherwise repeat Task 3 and restart the two-round count.

- [ ] **Step 5: 写入综合结论与 L1 接口**

The final report must state:

```text
what is already covered by direct priors
what appears distinct before candidate freeze
which distinctions cannot be judged until the exact recurrence exists
the strongest current rejection argument
the minimum L1 claim-by-claim novelty questions
the exact search cutoff timestamp
```

- [ ] **Step 6: 运行最终文档和数据检查**

Run:

```bash
python3 - <<'PY'
import csv, json
from pathlib import Path
log = [json.loads(line) for line in Path(".aris/literature/l0_search_log.jsonl").read_text().splitlines() if line.strip()]
with Path("docs/research-direction/12_optimizer_state_claim_matrix.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
report = Path("docs/research-direction/11_optimizer_state_literature_update.md").read_text(encoding="utf-8")
assert any(row["query_id"].startswith("S1-") for row in log)
assert any(row["query_id"].startswith("S2-") for row in log)
assert "candidate_not_generated" not in report
assert "## 饱和检查" in report
assert rows
print({"queries": len(log), "papers": len(rows), "status": "L0 complete"})
PY
git diff --check
```

Expected: a dictionary with `status: L0 complete`, followed by a clean `git diff --check`.

- [ ] **Step 7: 提交 L0 文献检索产物**

```bash
git add docs/research-direction/11_optimizer_state_literature_update.md docs/research-direction/12_optimizer_state_claim_matrix.csv
git commit -m "research: close L0 optimizer-state literature search"
```
