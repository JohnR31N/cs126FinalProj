# 五篇候选论文，哪篇最适合从复现走到研究？

**推荐 Concrete Autoencoders（CAE）。** 可以把题目定为 **《提前消除重复选择，能改善特征重建吗？》** 后续作者在两个任务中观察过独立选择槽位挑中同一输入的现象。我们又有一个值得解释的现成结果：惩罚提前消除了训练记录里的重复，但测试重建误差上升。由此可以设计一个只改变惩罚持续时间的实验。若希望做聚类，首选 IDEC，围绕初始化和训练平台期做对照。

以下按本课程从已有代码走到可解释对照实验的难度排序。课题的研究价值和实现难度分别考虑。三项本地 pilot 都用三个种子，数据、预算与原论文有差异。PVC 和 VaDE 在本仓库处于文献调研阶段。[本地实验协议](experiment_plan.txt) · [已测结果](../results/pilot/RESULTS.txt)

| 选题顺序 | 后人指出的具体问题 | 课程里可以控制什么 | 主要工作量 |
|---|---|---|---|
| 1 · CAE | 多个槽位可能选中同一列，训练期间出现重复与训练不稳定有关 | 固定特征数、初始化和温度曲线，仅改变去重惩罚的持续时间 | 低。已有逐轮抽样记录、选后回归和两个预算 |
| 2 · IDEC | 初始划分可能限制后续改进，只更新聚类中心有时很快回到原来的划分 | 从同一个预训练模型分叉，控制中心重算和网络权重的部分重置 | 中。能复用模型，需要区分平台期和步数不足 |
| 3 · Deep SVDD | 污染样本被当作正常样本拉近中心，过滤又可能误伤正常样本 | 控制污染比例及稀有正常样本占比 | 中。已有干净基线，需构造污染和分组评估 |
| 4 · PVC | 原模型面向两个视图，跨视图对齐依赖同时出现的样本 | 保持两个视图和总配对数相同，只改变各簇的配对覆盖 | 中高。先验证 MATLAB 基线、样本行对齐和缺失掩码 |
| 5 · VaDE | 变分推导有明确勘误，端到端成绩对预训练敏感 | 固定混合分布做数值检验，再比较初始化或后验估计 | 完整模型高。如果擅长概率推导，数学切口反而最清楚 |

## 能找到哪些 review

截至 2026-11-22，按原题名、作者、会议版本检索了会议论文页、公开网页索引和 OpenReview 相关记录。这五篇原投稿的匿名评审正文均未能核实。AAAI 原站在本次访问中限制了读取，PVC 正文使用作者版。OpenReview API 的题名查询也返回了 403。这是一次定向检索，有漏检可能。[IDEC 原文](https://www.ijcai.org/proceedings/2017/243) · [VaDE 原文](https://ijcai.org/proceedings/2017/273) · [PVC 作者版](https://www.lamda.nju.edu.cn/publication/aaai14pvc.pdf) · [CAE 原文](https://proceedings.mlr.press/v97/balin19a.html) · [Deep SVDD 原文](https://proceedings.mlr.press/v80/ruff18a.html)

**有一份公开评审能说明 IDEC 在后续研究中怎样被评价。** NeurIPS 2019 的 *Learning Representations for Time Series Clustering* 中，Reviewer 2 指出等长序列也能作为静态向量处理，要求加入静态深度聚类方法比较。作者回复称他们运行了 DEC 和 IDEC 的官方实现，并在表 1 给出 36 个时间序列数据集的 Rand Index。审稿人提出的是公平对照的要求，选择 IDEC 和报告分数的是后续论文作者。VaDE 后续论文 MFCVAE 的 NeurIPS 页面也列出了公开评审入口，本次访问论坛时遇到了验证页。[Reviewer 2 原文](https://proceedings.neurips.cc/paper/2019/file/1359aa933b48b754a2f54adb688bfa77-Reviews.html) · [作者回复与表 1](https://proceedings.neurips.cc/paper_files/paper/2019/file/1359aa933b48b754a2f54adb688bfa77-AuthorFeedback.pdf) · [MFCVAE 会议页面](https://proceedings.neurips.cc/paper_files/paper/2021/hash/48cb136b65a69e8c2aa22913a0d91b2f-Abstract.html)

综述适合定位方法，具体缺点最好追到做过对照的文章。Zhou 等的深度聚类综述把 IDEC 和 VaDE 放进不同的模型家族。Ruff 等的异常检测综述讨论单类网络如何约束表示以避免退化，其中有 Deep SVDD 原作者参与。《不完整多视图聚类综述》表 3 直接列出 PVC 的优缺点。CAE 的重复选择有更直接的后续实验，见下文。[深度聚类综述](https://doi.org/10.1145/3689036) · [异常检测综述](https://ml.cs.rptu.de/publications/2021/ieee.pdf) · [不完整多视图聚类综述，表 3](https://www.joca.cn/EN/10.11772/j.issn.1001-9081.2023060813)

## 1. Concrete Autoencoders

Nilsson 等在 ICML 2024 的 IP-CAE 中直接研究了 Balın 等的模型。图 1 将训练不稳定与重复选择联系起来，图 5 追踪选择数量和性能。他们比较了间接参数化和显式 GJS 多样性正则，表 1、2 分别报告重建和分类。更早的 Strypsteen 与 Bertrand 在 EEG 通道选择中也发现相同通道会被多个槽位选中，并用带阈值的惩罚限制同一通道累积的选择概率。EEG 的目标是下游任务表现，我们用的是选定像素后的线性重建。两篇都提供了去重的先例，两篇的正则形式也都与本仓库的概率内积惩罚有别。[IP-CAE，图 1、图 5、§2.5](https://proceedings.mlr.press/v235/nilsson24b.html) · [EEG 通道选择，§III-B、式 5](https://ftp.esat.kuleuven.be/pub/pub/sista/tstrypst/EEG_CS.pdf)

**重新读取现成的训练轨迹，能看到惩罚确实改变了选择过程。** 下表的 epoch 是日志中最后一次出现少于 50 个不同 argmax 像素的记录，三列按种子 0、1、2 排列。日志记录第 1 轮及每 20 轮，早期低置信度下的 argmax 可能频繁变化。MSE 是最终硬选择后重拟合 OLS 的测试均值。这个轨迹统计是事后分析。[800 轮逐种子记录](../results/pilot/cae) · [1600 轮逐种子记录](../results/budget1600/cae) · [两次预算的实验说明](budget_followup_plan.txt)

| 预算与方法 | 最后一次记录到重复的 epoch | 最终不同像素数，各种子 | 测试 MSE ↓ |
|---|---|---|---:|
| 800 · 原版 | 500 / 400 / 200 | 50 / 50 / 50 | 0.019084 |
| 800 · 持续惩罚 | 60 / 60 / 40 | 50 / 50 / 50 | 0.024513 |
| 1600 · 原版 | 640 / 280 / 480 | 50 / 50 / 50 | 0.018375 |
| 1600 · 持续惩罚 | 60 / 60 / 40 | 50 / 50 / 50 | 0.021110 |

在两个预算下，每个配对种子的惩罚版 MSE 都更高。加倍训练后，均值差距从约 28.4% 缩到 14.9%。1600 轮时原版三个种子都超过论文采用的 0.99 平均最大选择概率，惩罚版三个都未达到。现有结果可以支持“更早消掉重复和更好的最终重建可以分离”，惩罚改变收敛速度仍然是合理解释。[800 轮汇总](../results/pilot/summary.json) · [1600 轮汇总](../results/budget1600/summary.json)

**建议的 final 实验**：固定 50 个槽位、同一解码器、退火和数据划分。在 MNIST 与 Fashion-MNIST 各用 5 个配对种子，比较原版、全程惩罚、仅前 25% 训练使用同一强度惩罚。第三组是待检验修改。预先固定 1600 轮，主指标为测试 OLS MSE，辅助指标为硬选择重复数、平均最大概率和验证重建曲线。逐种子报告三组差值。另设相同上限的延长训练检查，记录达到 0.99 所需的步数；到上限还未达到的种子照实保留。若早停惩罚保留较少的早期重复，且重建误差回到原版水平，结果支持“惩罚持续时间影响重建”；若三组在相近置信度下差距缩小，收敛速度更值得解释。普通 MNIST 已参与形成假设，第二数据集承担新的检验。该项目的贡献是检验去重收益的条件。

## 2. IDEC

Guo 等保留重建项的动机，是防止优化聚类目标时损坏已有表示。Mrabah 等在 DynAE 中从另一个方向提出批评：重建每个输入也要求表示保留簇内差异，因此与压紧同簇样本的目标可能竞争。他们在文中标为 IDEC* 的对照上分析训练动态。Miklautz 等给了另一个可单独复现的现象：ICLR 2025 的 BRB 图 1 在 USPS 上比较 IDEC、周期性重算中心、部分重置权重后重算中心。单独重算中心仍遇到平台，联合操作才突破。两篇讨论的是不同机制，做 final 选其一即可。[IDEC 原文](https://www.ijcai.org/proceedings/2017/243) · [DynAE 作者全文](https://arxiv.org/pdf/1901.07752) · [BRB 图 1、§6.2](https://proceedings.iclr.cc/paper_files/paper/2025/file/f91685b940d5032f0f0c247edbd72edd-Paper-Conference.pdf)

本地 12,000 张训练图上，IDEC 的训练集聚类 ACC 为 74.94%，同初始化去掉重建项为 78.00%，预训练 AE 加 K-means 为 74.53%。三个 IDEC 种子都到达 4,000 步上限。最适合我们的是先延长相同预算并记录标签变化，确认是否存在稳定平台。之后用同一预训练模型做四组：原版、仅重算中心、仅部分重置、两者都做。固定触发轮次和总更新数，报告 ACC/NMI 及重算前后分配的一致性。只有重算中心后又回到相似分配、而联合重置能持续改变它时，才能把结果和 BRB 的解释联系起来。若时间有限，先做单一数据集和同一预训练条件。这个对照比同时改变重建权重、初始化和停止规则更容易解释。[本地 IDEC 记录](../results/pilot/idec) · [BRB 公开论文及消融设计](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f91685b940d5032f0f0c247edbd72edd-Abstract-Conference.html)

## 3. Deep SVDD

Ruff 等在原论文 §3 区分了平均距离损失与 soft-boundary 损失，也给出了退化解和网络约束。课程里更容易检验的批评是训练集污染。Qiu 等的 LOE 在 §4.1、图 1 做了直接对照：二维三成分高斯混合，10% 污染，使用带 RBF 的深度单类模型。把所有训练样本当正常会让异常区域也进入正常范围；过滤可部分改善，给潜在异常分配训练目标后边界更清楚。该文的图像实验使用其他骨干，不能将其图像分数当作原始 CNN Deep SVDD 的污染结果。[原文 §3](https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf) · [LOE §4.1–4.2、图 1](https://proceedings.mlr.press/v162/qiu22b/qiu22b.pdf)

本地正常类 0、3、8 的干净测试 AUROC 分别为 97.15%、89.91%、91.50%。如果做异常检测，我会问：**过滤大距离训练点对少见但正常的写法有什么代价？** 从干净训练集构造可追踪的少数正常子群，例如只对固定的 5% 正常图作轻度平移；另行混入 0%、5%、10% 的异类数字。在训练总量和优化步数匹配的条件下比较原版、固定比例截尾，另外记录污染为零时截尾本身的损失。以独立正常验证集校准阈值，分别报告总体 AUROC 和两种正常子群的误报率。子群身份仅用于构造和评估。这个方案多出污染、正常子群和阈值三项设计工作。[本地 Deep SVDD 结果](../results/pilot/svdd) · [LOE 对正常样本误判的讨论](https://proceedings.mlr.press/v162/qiu22b/qiu22b.pdf)

## 4. Partial Multi-View Clustering（PVC）

Li 等的结论主动说明只研究了两个视图，并指出更多视图会带来计算问题。Hu 与 Chen 在 DAIMC 的引言中直接评价 PVC 的两视图和非负数据限制，其改进使用加权 semi-NMF，能原生处理更多视图和负值。《不完整多视图聚类综述》表 3 认可 PVC 的聚类与收敛表现，并列出噪声敏感、两视图等局限。后续的 *Multi-view clustering on data with partial instances and clusters* 已进一步考虑某簇的全部样本都从一个视图消失的情况。这些工作使“把两个视图扩展到三个”作为新贡献的空间很小。[PVC 结论](https://www.lamda.nju.edu.cn/publication/aaai14pvc.pdf) · [DAIMC 引言](https://www.ijcai.org/proceedings/2018/0313.pdf) · [综述表 3](https://www.joca.cn/EN/10.11772/j.issn.1001-9081.2023060813) · [部分样本与部分簇研究](https://doi.org/10.1016/j.neunet.2020.05.021)

可做的窄问题是**总配对数相同时，把双视图配对集中在少数簇，会不会特别损伤其余簇？** 从完整的双视图数据生成随机缺失与簇条件缺失，保持各视图样本数、总配对数和单视图边际簇分布可比。先用已知标签的合成数据检查掩码和行对齐，再用一份真实数据。比较 NMI 和低覆盖簇召回率，标签仅用于造掩码和评估。首先要在均匀缺失下跑通并核对作者 MATLAB 实现，这是它排第四的实际成本。[作者代码](https://www.lamda.nju.edu.cn/code_PVC.ashx)

## 5. VaDE

Falck 等在 NeurIPS 2021 的 MFCVAE §3.1、定理 1 与附录 B.1.1 指出一个具体推导问题。VaDE 的分解后验对每个输入使用固定的簇分布 $q(c\mid x)$，而 $p(c\mid z)$ 会随采到的 $z$ 改变。两者 KL 的期望一般只能最小化到正值。最优的 $q(c\mid x)$ 正比于 $\exp(\mathbb{E}_{q(z\mid x)}[\log p(c\mid z)])$。作者解释了原文的实验结果为何仍然成立。这是五篇中最容易精确指到一条公式的批评。[VaDE 原文](https://ijcai.org/proceedings/2017/0273.pdf) · [MFCVAE §3.1、定理 1](https://proceedings.neurips.cc/paper_files/paper/2021/file/48cb136b65a69e8c2aa22913a0d91b2f-Paper.pdf) · [附录 B.1.1](https://proceedings.neurips.cc/paper/2021/file/48cb136b65a69e8c2aa22913a0d91b2f-Supplemental.pdf)

另一篇 VIB-GMM 在 MNIST 上各运行 10 次。其表 1 的无预训练 VaDE 最佳聚类 ACC 为 91.8%，平均 **78.8%（标准差 9.1 个百分点）**。表 2 使用预训练权重时平均 **93.2%（标准差 1.5 个百分点）**。这给出了把预训练作为实验因素、保留所有种子的具体理由。[VIB-GMM §4.3、表 1–2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516645/)

偏理论的团队可以先用两个高斯分量、固定编码器后验，算出最优簇责任度和最小期望 KL，再比较单样本、多样本估计与数值积分基准。为了满足课程要求，还需要将估计方法接入可运行的 VaDE 核心，固定初始化、观测训练代价和聚类结果。已有论文给出了正确推导，课程贡献可以是说明近似误差在什么后验宽度下会影响实际训练。当前仓库尚缺 VaDE 基线，若更关心按时完成实证项目，我会选 CAE 或 IDEC。
