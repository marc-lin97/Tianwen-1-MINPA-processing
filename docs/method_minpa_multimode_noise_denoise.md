# Tianwen-1/MINPA 多模式噪声评估与去噪方法

## 版本和状态

本方法包含两个明确隔离的版本层级：

- `minpa-paper-channel-denoise-v1.1.0`：已冻结的 Mode 1 生产基线。模型包含 175 个批准条目、去重后 153 个独立区间、4,350 条记录；模型 SHA-256 为 `9d4030196841343ab034c1418ec3243f052c2cff0bc2b87dd7c90bc604a74b24`。本次扩展不重建、不覆盖该文件。
- `minpa-multimode-noise-review-v2.0.0-rc1`：Mode 4/12 候选发现和人工审查框架。rc1 只能产生 `provisional_review_only` 诊断，不是生产背景模型。
- `minpa-multimode-channel-denoise-v2.0.0`：预留的正式去噪策略版本。只有人工审查完成、模型验证通过并冻结模型及 bundle 哈希后才能发布。

默认 `background_policy="none"` 始终不变。若 rc1 审查失败、模型验证失败或 bundle 哈希不匹配，回退行为是继续使用冻结的 Mode 1 v1.1.0，并保持 Mode 4/12 原始值。

## 数据量纲和模式布局

本地 `Ion_Count` 按已发布差分粒子通量 DPF 解释，单位为 `1/(s cm² sr eV)`，不宣称它等于论文中的原始探测器 counts。所有估计和扣除均在仪器通道空间进行，统一维序为 `energy × pitch × azimuth × mass`。

| 模式 | 单个处理时刻布局 | 说明 |
|---|---|---|
| Mode 1 | `40 × 4 × 16 × 8` | 论文法已由本项目人工标定并冻结 |
| Mode 4 | `64 × 4 × 16 × 8` | 项目对论文法的扩展 |
| Mode 12 | `48 × 1 × 1 × 32` | 每条原始记录先拆为 `t+1.025 s` 和 `t+3.075 s` 两个子记录 |

Mode 12 是方位积分的单角度产品，不能恢复方向依赖背景、三维流速或二维 VDF。论文并未验证 Mode 4/12；这里使用相同估计思想属于项目方法扩展。

## rc1 安静候选发现

扫描本地 2021–2025 年 `day_spe` 和只读 `ori` 数据。H+、O+、O2+ 分别执行下列步骤，审批也彼此独立：

1. 对每个 `mode/species/month` 建立 UTC 对齐、互不重叠的 60 秒基础窗。
2. 要求对应物种项目质量存在，质量位 3–4（掩码 `0x0C`）不出现；原生 `Quality` 必须等于零。ori 中字符型十六进制值（例如 `0x00`）必须先按十六进制解码，不能把 ASCII 字符码当成质量值。
3. 记录覆盖率至少 80%，最大间隔不超过标称 cadence 的 1.5 倍。Mode 4 cadence 为 12.3 s，Mode 12 子记录 cadence 为 2.05 s。
4. 计算 `median_t(sum_E(max(DEF(E,t),0)))`。
5. 保留月内最低 10% 的合格基础窗，位于分位阈值的并列值全部保留。
6. 合并同月、同模式、同物种且时间连续的通过窗。最短 60 秒，不设最大时长、数量上限或自动分数淘汰。

预检还要求被选候选存在对应 ori，并核验原始数组长度：Mode 4 每条 32,768 值；Mode 12 每条原始记录 3,072 值，拆分后每子记录 1,536 值。任何被选候选缺少源文件、能量表不匹配或布局错误，都会阻止正式索引发布。

## 人工审查工作区

全部 PNG 平铺在 `outputs/minpa_multimode_noise_review_v2.0.0-rc1/pending/`，没有年份、月份、模式或物种子目录。文件名为：

```text
mode04_Hplus_<candidate-id>_<start>_<stop>.png
mode12_Oplus_<candidate-id>_<start>_<stop>.png
```

每张图只对应一个模式、一个物种和一个区间。图中包含候选前后各 12 分钟的原始能量—时间谱、黑色起止线和候选平均能谱；质量不合格记录留白，不平滑、不插值。每个 `mode/species` 使用统一 LogNorm 色标，色限为其质量合格正值样本的 1%–99.7% 分位，实际色限和确定性采样规则写入 manifest。

审查结束后：保留 PNG 表示批准该物种区间，删除表示拒绝。意外改名、额外 PNG 或嵌套 PNG 会使终结器报错。整窗删除不会自动裁边、拆分或挽救；即使拒绝原因只是部分区间包含有效观测，也保持整窗拒绝。

## 逐通道背景估计

对物种 `s` 的人工批准区间 `j` 和该物种覆盖的通道 `c`：

\[
b_{j,c}=\operatorname{mean}(x_{t,c}\mid x_{t,c}>0),\qquad
B_c=\operatorname{mean}_j(b_{j,c}).
\]

区间之间等权，不按时长、记录数或自动分数加权。不同物种只使用各自批准清单：

- Mode 4：H+、O+、O2+ 分别只处理 1、16、32 amu；
- Mode 12：H+ 使用 0.76/1.25 amu，O+ 使用 15.56/16.45 amu，O2+ 使用 30.94/33.17 amu；
- 其他质量通道为未审通道，保持原值，不使用最近物种或跨质量代理。

每通道保存背景 DPF、批准区间支持数、总/正值样本数、零值率、正值率、案例间标准差与标准误、非零中位数/MAD，以及固定种子 5,000 次区间级 bootstrap 95% CI。样本不足时不伪造 CI。

支持等级为：

- `unsupported`：无正值样本，保持原值；
- `low_support`：少于 3 个批准区间或少于 5 个正值样本，执行扣除并标记；
- `supported`：至少 3 个批准区间且至少 5 个正值样本。

## 去噪公式和生产接口

正式模型的逐通道扣除为：

\[
x^{\mathrm{corrected}}_{t,c}=\max(x_{t,c}-B_c,0).
\]

NaN 保持 NaN，禁止平滑、插值和跨质量补值。Mode 12 必须在双子记录拆分之后扣除。未来显式策略名为 `multimode-paper-channel-subtract`；它按模式从冻结 bundle 加载模型，并拒绝 rc1、未完成人工审查、无效或哈希不匹配的模型。原有 `paper-channel-subtract` 继续只读取冻结 Mode 1 路径；Mode 7 始终保持原值。

## 可复现入口和产物

- 配置：`config/minpa_multimode_noise_review_v2.0.0-rc1.json`
- 候选与制图：`scripts/prepare_minpa_multimode_review_workspace.py`
- 文件存在性终结器：`scripts/finalize_minpa_multimode_review_from_files.py`
- 通用估计器：`src/highE/minpa_multimode_background.py`
- 候选规则：`src/highE/minpa_multimode_review.py`
- 审查输出：`outputs/minpa_multimode_noise_review_v2.0.0-rc1/`

正式索引只在所有候选图成功生成后原子写入。运行中断时只保留检查点和已生成图片，恢复运行会核对既有文件名并跳过已完成 PNG，不会把不完整目录标记为正式工作区。manifest 保存配置哈希、源文件路径/SHA-256/修改时间、布局、质量策略、选择阈值、统一色限、基准结果和算法版本。

## 适用与不适用范围

适用：Mode 1 冻结模型；人工审查并正式冻结后的 Mode 4/12 指定物种质量通道；以发布 DPF 为输入的逐通道背景评估和零截断扣除。

不适用：Mode 7；Mode 4/12 未审质量通道；rc1 自动候选的生产扣除；原始 detector counts 的恢复；UV/奇偶能道错误修复；有效信号与背景不可分辨的混合窗自动裁边；Mode 12 方向背景、三维速度或二维 VDF 重建；跨模式、跨物种或跨质量通道插值。
