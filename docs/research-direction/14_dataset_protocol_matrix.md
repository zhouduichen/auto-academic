# 数据集协议、许可与数据门矩阵

> 审计日期：2026-07-31
> 状态含义：`ready` 表示来源和合法数据门可预注册，不表示已授权下载或运行；`conditional`
> 需要先解除列出的条件；`reject-for-now` 当前不能形成可信、合法或单卡可承受的数据门。
> 强约束：本文件不打开、不下载、不检查任何未来 final test 内容。

## 1. 排序结论

| 优先级 | 数据集 | 状态 | 适合作用 | 现在不做什么 |
|---:|---|---|---|---|
| 0 | CIFAR-100 + 40% synthetic instance-dependent noise | `active/frozen` | 当前 M1 唯一数据门 | 不构造官方 test loader；不改变 40k/2.5k/2.5k/5k 分区 |
| 1 | CIFAR-10N / CIFAR-100N | `ready-after-M1-GO` | R3 小规模 human-label-noise 复制 | 不提前选择 10N/100N label set，不下载，不看 test |
| 2 | Food-101N | `conditional` | 更大视觉域的真实网络噪声与干净 Food-101 final test | 不使用人工验证标记训练候选；先确认存储、下载条款和单卡 wall-clock |
| 3 | FSDnoisy18k | `conditional/optional` | 只有跨模态主张必要时的 R6 音频外部验证 | 不把 `manually_verified` 标记交给训练算法；M1 前不启动 |
| 4 | Clothing1M | `reject-for-now` | 经典真实噪声比较 | 未找到可核验的官方活跃下载与数据许可，不能先从镜像获取 |
| 5 | WebVision 1.0 | `reject-for-now` | 大规模网页噪声压力测试 | 约 44 GB resized 数据且 2.4M 图像；条款禁止再分发，单 5080 成本不适合 R3 首站 |

此排序只准备数据门。M1 `GO` 后仍须依据 M1 效应、方差、失败模式和最终主张另写 R3 设计；
不能因为 `ready` 就自动提交 Windows。

## 2. 当前 M1 数据门（绑定）

数据为 CIFAR-100，模型为 `google/vit-base-patch16-224-in21k` 的 rank-8 LoRA。split seed
`20260801`：40,000 train、2,500 tuning、2,500 development、5,000 confirmation；官方
10,000 test 永不构造或加载。40% instance-dependent noise 只由 40,000 train 生成，恰好
16,000 个标签改变且 noisy label 必须不同于 clean label。

必须保存但不得给候选读取：corrupted IDs、clean/noisy labels、propensity、transition
probabilities、feature extractor digest 与 generator digest。任何 tuning/development/confirmation/
test 信息进入噪声生成都会使整个 gate 失效。

## 3. 真实噪声候选详表

### 3.1 CIFAR-10N / CIFAR-100N — `ready-after-M1-GO`

- **官方来源：** [UCSC-REAL 作者仓库](https://github.com/UCSC-REAL/cifar-10-100n)，审计
  commit `49df7d8a69e355470c77c1c2f2424916325a394b`；论文发表于 ICLR 2022。
- **label provenance：** 对 CIFAR train 图像进行人类重标。CIFAR-10N 提供多组 noisy label；
  CIFAR-100N 提供一组。不能事后挑对候选最有利的一组。
- **License：** 仓库数据许可为 CC BY-NC 4.0；只用于符合条款的研究，产物不得重新分发原图或标签包。
- **关键顺序风险：** 官方仓库明确提供 `image_order_c10.npy` / `image_order_c100.npy`，因为
  TensorFlow Datasets 与 PyTorch CIFAR 顺序不同。sample ID 必须以官方映射和原始文件 hash
  固定，绝不能按“看起来相同的数组下标”对齐。
- **单卡风险：** 小；与 M1 同量级。但它与合成 CIFAR-100 共享数据家族，所以只能支持
  “human-label-noise replication”，不能单独支持跨域或规模主张。

候选数据门（仅模板，须在 R3 设计中二选一并冻结）：

1. 从 50k noisy train 按 noisy label 分层产生 40k train、5k tuning、5k development；
2. train 只暴露 noisy label；tuning/development 可使用原 CIFAR label 作离线评价，但训练算法
   不得读取；
3. 方法、超参数、主要比较和分析代码冻结后，官方 10k test 仅打开一次作为 final；
4. clean train label 仅用于 gate 评价和污染诊断，必须以访问日志证明未进入 loss、detector、
   scheduler 或 admission；
5. CIFAR-10N 的具体 label set 或 CIFAR-100N 必须在下载前按科学问题冻结，不能并行试后择优。

### 3.2 Food-101N — `conditional`

- **官方来源：** [Food-101N 项目页](https://kuanghuei.github.io/Food-101N/)。
- **规模与噪声：** 310,009 张图、101 类，项目页估计约 20% noisy labels；提供 52,868 个
  verified training labels 和 4,741 个 verified validation labels。
- **评价：** final 使用原 Food-101 clean test；verified metadata 是可能泄漏 clean/noisy 状态的
  side information，不得作为候选输入、sample weight、threshold 或训练过滤信号。
- **条款：** 仅限非商业研究；项目页明确不授予底层图片的额外知识产权。发布 artifact 只能保存
  URL/ID/hash/统计，不再分发图像。
- **单 RTX 5080 风险：** 中等。图像量约为 CIFAR train 的 6 倍且分辨率高；在下载前要用
  M1 的实测吞吐做 epoch/step/wall-clock/磁盘估算，冻结预算后才可进入 R3 或 R4。

候选门：noisy images 作 train；将 4,741 verified validation 预先哈希分成 tuning/development
两部分；52,868 verified-train labels 封存为冻结后的诊断层，不供方法选择；Food-101 clean test
为 test-once final。若 verified validation 太小，宁可降低主张或放弃该集，不能反复查看 final test。

### 3.3 FSDnoisy18k — `conditional/optional`

- **官方来源：** [Zenodo record](https://zenodo.org/records/2529934) 与
  [作者项目页](https://www.eduardofonseca.net/FSDnoisy18k/)。
- **规模：** 18,532 clips、约 42.5 小时、20 类、约 9.5 GB；train 17,585，test 947 且人工验证。
- **噪声来源：** Freesound 用户标签；train 中同时有 verified 和 non-verified subset，并提供
  `manually_verified` 字段。
- **License：** 数据集实体 CC BY；单条音频为 CC BY 或 CC0，artifact 必须保留逐条 attribution。
- **泄漏风险：** 候选如果读取 `manually_verified`，等同获得 clean-reference signal，直接违反
  方法合同。loader 必须在算法可见 schema 中删除该字段。
- **角色：** 只有视觉 R2–R5 已通过且论文确实需要 C7 跨模态主张时才启动。音频 front-end、
  backbone、训练预算与视觉不同，不能把失败解释为视觉方法失效，也不能因成功反推普适性。

候选门：从 train 中预先留出不重叠的 tuning/development，并只由独立数据管理员在评价层使用
verification 信息；剩余 train 对算法只呈现 clip、class label 和匿名 ID；官方 clean test 为
test-once final。若无法技术性隔离 verification flag，则拒绝该数据集。

### 3.4 Clothing1M — `reject-for-now`

- **原始来源：** [CVPR 2015 论文](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf)。
- **规模：** 约 1M noisy train、14 类；论文描述 72,409 张人工精修 clean subset，经典协议将其
  分为 clean train/validation/test。
- **阻塞：** 本轮未找到可核验、活跃且带明确数据许可的作者官方下载入口。第三方云盘、Kaggle
  或代码仓库镜像不能替代数据权利审计。
- **恢复条件：** 找到作者/机构原始分发页及 terms，固定 archive hash，并能形成 noisy train、
  clean tuning、clean development、clean final 的不可交叉门。否则不下载、不运行、不引用
  “我们在 Clothing1M 复现”。

### 3.5 WebVision — `reject-for-now`

- **官方来源与条款：** [ETH Zürich WebVision 下载页](https://data.vision.ee.ethz.ch/cvl/webvision/download.html)。
  页面要求研究用途并禁止数据再分发，底层图片版权责任由使用者承担。
- **规模：** WebVision 1.0 约 2.4M 图、1,000 类；官方 resized 下载约 Flickr 26 GB、Google
  16 GB，加 validation/test 约 1.6 GB。2.0 分片更大，不在当前范围。
- **计算风险：** 单 RTX 5080 的 ViT-LoRA 完整多 seed 调参与 confirmation 远超 R3 小规模复制；
  同时下载、坏链、图像去重和 ILSVRC/WebVision evaluation 对齐均增加审计负担。
- **恢复条件：** 只有 R3 成功、C4 规模主张仍必要，并有按 M1 实测吞吐形成的独立 R4 预算与
  存储清单后，才审计 WebVision 1.0；不得用缩小到任意子集后仍声称“大规模 WebVision”。

## 4. 通用数据门合同

### 4.1 打开顺序

```text
来源/terms/hash 审计
  -> sample-ID 与 label provenance 固定
  -> tuning/development/final ID 哈希提交
  -> 方法、参数、主要比较、样本量、分析代码冻结
  -> tuning（可反复但完整留痕）
  -> development（只打开一次，用于 GO/NO-GO）
  -> final（完整矩阵结束后只打开一次）
```

“只打开一次”指评价脚本首次解封后不得修改方法、超参数、checkpoint 选择或主要终点；若数据/
配对/代码完整性失败，整个门作废，修复只能用未使用的新门，而不是重跑同一个 final。

### 4.2 必须写入 manifest 的字段

- dataset canonical name、version、official URL、terms snapshot hash、archive SHA-256；
- 原始 sample ID、内容 hash、label source、split role、是否 human-verified；
- split seed/algorithm、分层字段、每类计数、重复/近重复检查；
- loader/framework 顺序映射与映射文件 hash；
- 算法可见字段白名单和禁止字段访问日志；
- tuning/development/final 首次访问时间、调用栈摘要、source/config hash；
- 每 run 的实际 examples seen、丢失/损坏样本、replacement policy；
- final prediction 文件和评价脚本 hash。

### 4.3 泄漏与偏差检查

- clean/verified 标记只存在于隔离评价表；训练进程读取即 fail closed。
- 网页数据的重复图、同主体/近重复跨 split、失效 URL 和 class prior 必须报告。
- 困难 clean、少数类和系统性偏置不能用总体准确率掩盖；预注册 worst-class、macro-F1、
  balanced accuracy 和 hard-clean 指标。
- 真实噪声数据并不自动代表部署噪声。每个主张必须限定 label provenance、域和收集过程。
- 不把 image、clip、batch、epoch 或 checkpoint 当独立统计重复；实验单元仍是配对 seed。

## 5. 启动前置条件

任何新数据集 GPU 任务必须同时具备：M1 `GO`；对应 C2/R2 结果未触发硬停止；合法来源和
terms；冻结三门 ID；预注册主要比较与最小效应；按 `16_post_m1_statistics_contract.md` 计算的
seed 数；基于 M1 实测吞吐/VRAM 的 Windows 预算；用户批准的新设计。缺一项均不提交。
