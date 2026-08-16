# MINPA Mode 4/12 逐通道背景与去噪 v2.0.0 发布说明

## 发布结论

`minpa-multimode-channel-denoise-v2.0.0` 已冻结为 Mode 4/12 的正式、哈希校验 bundle。模型只覆盖人工逐物种批准的质量箱；Mode 7、未审质量箱和无支持通道保持原值。

本发布不改变 Mode 1 `minpa-paper-channel-denoise-v1.1.0`，其模型 SHA-256 仍为 `9d4030196841343ab034c1418ec3243f052c2cff0bc2b87dd7c90bc604a74b24`。

## 最终人工审核

权威清单为 `outputs/minpa_multimode_noise_review_secondary_prefilter_v2/manual_review_final.json`，SHA-256 为 `089993e253ba089a4e6201c0062f03c7e8a6a324d0698372111eb910a9ba0821`。

5,120 个 5–10 分钟候选中，人工保留 2,901 个、删除 2,219 个。没有额外、改名、嵌套或被修改的 PNG。

| 模式/物种 | 批准 | 拒绝 |
|---|---:|---:|
| Mode 4 H+ | 125 | 234 |
| Mode 4 O+ | 564 | 665 |
| Mode 4 O2+ | 1,451 | 578 |
| Mode 12 H+ | 58 | 42 |
| Mode 12 O+ | 248 | 587 |
| Mode 12 O2+ | 455 | 113 |

批准条目覆盖2021–2025年、593个 day_spe 和1,378个 ori 文件。源文件总量约10.08 GiB，完整路径、大小、修改时间和 SHA-256 保存在 `source_inventory.json`；规范化源清单哈希为 `1bc060b372d9d48a4e013d2e4079b6423baebf09c59cde19474cd2586d361d05`。

## 背景估计与扣除

输入 `Ion_Count` 按已发布差分粒子通量 DPF 解释，单位为 `1/(s cm² sr eV)`，不宣称恢复原始探测器 counts。维序固定为 `energy × pitch × azimuth × mass`。

对物种 `s` 的批准区间 `j` 和审核通道 `c`：

\[
b_{j,c}=\operatorname{mean}(x_{t,c}\mid x_{t,c}>0),\qquad
B_c=\operatorname{mean}_j(b_{j,c}).
\]

区间等权，不按时长、记录数或筛选分数加权。Mode 12 在 `t+1.025 s` 和 `t+3.075 s` 双子记录拆分后估计和扣除。正式扣除为：

\[
x^{\mathrm{corrected}}_{t,c}=\max(x_{t,c}-B_c,0).
\]

NaN 保持 NaN，不平滑、不插值、不跨质量箱补值。`unsupported` 通道保持原值；`low_support` 通道执行扣除并标记；`supported` 表示至少3个批准区间且至少5个正值样本。

## 逐通道支持度

Mode 12 每个物种审核96个通道，全部为 `supported`。Mode 4 每个物种审核4,096个角度—能量通道：

| 物种 | supported | low_support | unsupported | 有限背景比例 |
|---|---:|---:|---:|---:|
| H+ | 1,815 | 1,568 | 713 | 82.59% |
| O+ | 516 | 1,715 | 1,865 | 54.47% |
| O2+ | 1,197 | 1,740 | 1,159 | 71.70% |

Mode 4 方向分辨率高且许多角度通道长期为零，尤其 O+ 的严格 `supported` 覆盖较低。这是本发布最重要的使用限制。不得通过邻近角度、质量箱或拟合曲面填补 `unsupported` 通道。

背景正值中位数约为：Mode 4 H+/O+/O2+ 分别 `1.61e4/1.35e4/1.60e4` DPF；Mode 12 分别 `3.88e3/3.94e3/4.13e3` DPF。量级与人工审核图中原始 DPF 的模式差异一致；模型没有跨模式使用绝对幅度阈值。

## 模型和 bundle 哈希

| 产物 | SHA-256 |
|---|---|
| Mode 4 模型 | `7900124ee156d93d0c72dc6d1f1e3004acffd5079154156f7dd1fc0c6dd2dc8b` |
| Mode 12 模型 | `6b0a852c6323535b5adab6e3cacf62a04de25ee0487270d74b7366d4a63a3c68` |
| bundle.json | `46e51b81ea19d3d75ae13043e9a917e6e197e9b1be3013fa76acac4513ecd678` |

发布入口为 `outputs/minpa_multimode_channel_denoise_v2.0.0/bundle.json`。生产策略名称为 `multimode-paper-channel-subtract`。加载器拒绝非 `frozen` 状态、模型哈希不匹配、模式键不匹配或未 finalized 的模型。默认策略仍为 `none`。

## 验证结果

确认通过：

- Mode 4 `64×4×16×8`、Mode 12 `48×1×1×32`、mass-fastest 重排和 Mode 12 双子记录时间；
- 1,378个源文件 SHA-256、原生 `Quality==0`、原始数组长度和每区间记录数；
- 模型保存/读取与 frozen bundle 哈希往返；
- 校正值非负且不高于原值；
- Mode 7、未审质量箱和无支持通道保持原值；
- 六个 Mode 4/12 × H+/O+/O2+ 的20倍背景合成强峰均保留至少95%，峰位移动不超过一个能道。

观测强峰验证为证据不足：从每个模式/物种选择12个高信号人工拒绝窗，共72例；其中没有审核通道达到“原始值/对应背景≥20”的验收阈值。因此不能宣称观测强峰验收通过或失败。72例的校正总量均未高于原始总量。完整记录见 `validation.json` 和 `release_manifest.json`。

## 性能与复现

100文件基准中，单进程26.61秒、双进程13.40秒，速度提高1.99倍；数组形状和内容哈希完全一致，因此正式提取使用2进程。1,378个源文件按文件生成可恢复分块，每个文件只读取一次。

构建环境：Python 3.12，NumPy 2.3.5，SciPy 1.18.0，h5py 3.16.0，Matplotlib 3.11.1，pytest 9.1.1。完整回归结果为150项通过。

```powershell
python scripts/finalize_minpa_multimode_review_from_files.py --workspace outputs/minpa_multimode_noise_review_secondary_prefilter_v2
python scripts/build_minpa_multimode_background_v2.py
python scripts/validate_minpa_multimode_background_v2.py
pytest -q
```

`outputs/minpa_multimode_channel_denoise_v2.0.0_work/` 保存约2.53 GiB的可恢复分块和统计池，以避免中断后重新扫描10.08 GiB源文件。
