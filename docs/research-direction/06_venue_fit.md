# 06 — 投稿适配分析 (Venue Fit)

> 状态：已完成 | 更新日期：2026-07-29
> ⚠️ CCF/T 等级需按投稿当日官方目录逐项核验，以下分析基于 2025 年已知目录

---

## 各选题的推荐投稿去向

### 选题 1（预算感知 PEFT Agent）→ 方法/系统论文

| 级别 | 会议 | 理由 | 风险 |
|---|---|---|---|
| **CCF-B** | **ECAI** | 欧洲 AI 旗舰；接受系统+实证论文；PEFT 方向适配 | 两年一届（偶数年），2026 是窗口 |
| **CCF-B** | **ICME** (IEEE) | 多媒体处理；PEFT 视觉应用适配 | 需强调视觉/多媒体场景 |
| **CCF-C** | **IJCNN** (IEEE) | 神经网络旗舰；接受 AutoML 系统论文 | 级别偏低，可作保底 |
| **CCF-B** | **COLING** | NLP 旗舰（如果扩展到多模态 LLM） | 纯视觉可能不匹配 |
| **CCF-B 期刊** | **Neural Networks** (Elsevier) | 神经网络；接受系统性方法论文 | 审稿周期长 |
| **非 CCF** | **AutoML Conference** | 领域最匹配，但不计 CCF | 可作 workshop 或补充投稿 |

### 选题 2（PEFT 稳定性基准）→ 基准/诊断论文

| 级别 | 会议/期刊 | 理由 | 风险 |
|---|---|---|---|
| **CCF-A** | **NeurIPS D&B** | 数据集与基准赛道；诊断性研究受鼓励 | 竞争激烈，需极其严谨 |
| **非 CCF** | **TMLR** | 接受诊断/负面结果；开放获取 | 不计入中国 CCF 目录 |
| **CCF-C** | **ICPR** (IAPR) | 模式识别；接受基准论文 | 级别偏低 |
| **CCF-B 期刊** | **Pattern Recognition** (Elsevier) | 接受系统评估论文 | 审稿慢 |

### 选题 3（Agent vs BO 比较）→ 实证/方法学论文

| 级别 | 会议/期刊 | 理由 | 风险 |
|---|---|---|---|
| **非 CCF** | **TMLR** | 最适合诊断性公平比较的期刊 | 不计入 CCF |
| **CCF-B 会议** | **ECAI** | 接受方法论论文 | 需包装为正面贡献 |
| **ICBINB @ ICLR** | Workshop | 专门接受负面/意外结果 | workshop，非正式发表 |

### 选题 4（SysML v2 数据集）→ 数据集论文

| 级别 | 会议/期刊 | 理由 | 风险 |
|---|---|---|---|
| **CCF-A** | **NeurIPS D&B** | 数据集与基准赛道；首个 SysML v2 数据集新颖性强 | 竞争激烈 |
| **CCF-C 会议** | **LREC-COLING** | 语言资源与评测；接受标注数据集 | SysML 太工程化可能不匹配 |
| **非 CCF** | **Scientific Data** (Nature) | 数据描述期刊；高影响力 | 需特别严谨的文档 |

---

## CCF 推荐目录参考

以下为当前已知的部分 CCF 推荐目录会议/期刊等级（仅供参考，必须逐项核验）：

### 会议

| 简称 | 全称 | CCF 等级 | 领域适配 |
|---|---|---|---|
| AAAI | AAAI Conference on AI | A | 通用 AI |
| IJCAI | Int'l Joint Conf. on AI | A | 通用 AI |
| NeurIPS | Neural Information Processing Systems | A | ML |
| ICML | Int'l Conf. on Machine Learning | A | ML |
| CVPR | Computer Vision and Pattern Recognition | A | CV |
| ICCV | Int'l Conf. on Computer Vision | A | CV |
| ECCV | European Conf. on Computer Vision | B | CV |
| ECAI | European Conf. on AI | B | 通用 AI |
| EMNLP | Empirical Methods in NLP | B | NLP |
| COLING | Int'l Conf. on Computational Linguistics | B | NLP |
| ICME | Int'l Conf. on Multimedia and Expo | B | 多媒体 |
| ICASSP | Int'l Conf. on Acoustics, Speech, and SP | B | 信号处理 |
| IJCNN | Int'l Joint Conf. on Neural Networks | C | 神经网络 |
| ICPR | Int'l Conf. on Pattern Recognition | C | 模式识别 |
| ICONIP | Int'l Conf. on Neural Information Processing | C | 神经网络 |
| PRICAI | Pacific Rim Int'l Conf. on AI | C | 通用 AI |

### 期刊

| 简称 | 全称 | CCF 等级 | 注 |
|---|---|---|---|
| TPAMI | IEEE Trans. PAMI | A |  |
| IJCV | Int'l Journal of Computer Vision | A |  |
| TIP | IEEE Trans. Image Processing | A |  |
| JMLR | Journal of Machine Learning Research | A |  |
| AIJ | Artificial Intelligence | A |  |
| PR | Pattern Recognition | B |  |
| Neural Networks | Neural Networks | B |  |
| Neurocomputing | Neurocomputing | C |  |
| PRL | Pattern Recognition Letters | C |  |
| TMLR | Trans. on Machine Learning Research | 非 CCF | 开放获取 |
| Machine Learning | Machine Learning (Springer) | B |  |

---

## 投稿策略建议

### 主攻路线（选题 1）

1. **第一目标**：ECAI 2026（CCF-B，偶数年窗口）或 ICME 2027
2. **同时准备**：AutoML Conference 2026/2027（不计 CCF 但领域匹配，可作论文基础）
3. **保底**：IJCNN 2027（CCF-C）

### 备选路线（选题 2）

1. **第一目标**：NeurIPS 2026 D&B Track（CCF-A，冲高）
2. **保底**：TMLR（滚动投稿）或 Pattern Recognition（CCF-B 期刊）

### 副线（选题 4）

1. **第一目标**：NeurIPS 2026 D&B Track
2. **保底**：Scientific Data 或 CCF-C 期刊

### 组合策略（如果资源允许）

**同一核心贡献可以产出两篇定位不同的论文**：
- **论文 A（方法）**：预算感知 PEFT Agent 的系统设计与验证 → ECAI/ICME（CCF-B）
- **论文 B（基准）**：PEFT 稳定性与方差基准 → NeurIPS D&B（CCF-A）或 TMLR

两篇共享实验基础设施（VTAB-1K 测试平台、PEFT 方法库）但贡献不同。

---

## 风险提示

1. **CCF 等级变动**：目录每年更新，投稿前必须逐项核验最新官方目录
2. **学院认定差异**：部分学院的"认可目录"与 CCF 目录不完全一致，需对表查证
3. **导师合著要求**：如果学院对作者顺序/第一署名单位有特殊计分规则，记录但不自行解读
4. **学位点要求**：部分学校对学位论文发表成果有期刊/会议的特殊规定（如要求期刊论文）
