---
document_id: tw1-minpa-spectrum-moments-algorithm
title: 天问一号 MINPA 从初始粒子数据到能谱与各阶矩的算法说明
version: 1.0
status: current-project-method-summary
updated: 2026-07-15
language: zh-CN
primary_instrument: Tianwen-1 MINPA
primary_species: [H+, O+, O2+]
coordinate_systems: [MINPA/instrument, orbiter-body, MSO, MSE]
authoritative_production_products: [day_spe, NV_MSO, NV_MSO_2, all-energy MSE grid archive]
reference_direct_product: raw-to-spectrum-moments Python processor
---

> 2026-07-17 correction: the current self-contained Python processor uses
> per-channel logarithmic energy edges (`dE_i=E_high,i-E_low,i`), not fixed
> `dE/E=0.15`. The signed permutation `[-V2,+V3,-V1]` maps MINPA payload
> components into spacecraft-body coordinates; MOMAG attitude then maps body
> coordinates to MSO. Sections describing fixed 15% widths refer only to the
> historical `NV_MSO` chain.

# 天问一号 MINPA 从初始粒子数据到能谱与各阶矩的算法说明

> 本文档面向研究人员和语言模型，按“输入—几何—物种—能谱—矩—坐标—质量—产品”的顺序描述当前项目算法。凡是生产链、参考实现和已知限制，均分别标明，避免把不同处理支路的假设混为一谈。

## 0. 给 ChatGPT 的读取约定

- 把本文档视为当前仓库截至 2026-07-15 的算法口径，而不是 MINPA 官方定标文件。
- `Ion_Count` 在本项目中解释为微分粒子通量 DPF：`cm^-2 s^-1 sr^-1 eV^-1`。能谱中的 DEF 由 `DPF × E` 得到。
- 当前成熟生产产品 `NV_MSO_2` 只保存零阶矩密度和一阶矩体速度；它不保存标量温度、压力张量或热流。
- 独立 Python 直算参考链可由原始通量计算物种能谱、密度、体速度和标量温度，但它不是 `NV_MSO_2` 的直接生成器。
- `NV_MSO_2` 的密度由旧 `NV_MSO` 密度按模式能量宽度缩放，速度则原样继承。不要声称它对每条记录重新积分了原始三维分布。
- 任何跨时刻统计都应先得到“每条记录的局地密度”，再做均值或中位数。跨时刻求和只能作为审计量，不能称为局地密度。
- mode 12 是特殊路径：当前质量标志生产链把每个拆分子记录解释为 `32 mass × 48 energy × 1 angle = 1536`。不能无条件按普通模式的 16 方位角重排。
- 天问一号全能级 MSE 网格产品固定使用 `Rm = 3397 km`；仓库其他通用常量中出现的 `3389.5 km` 不适用于该归档。

## 1. 当前算法的总体结构

当前项目存在三条相关但不完全相同的处理支路。

| 支路 | 输入 | 能量范围 | 物种 | 主要输出 | 当前角色 |
|---|---|---|---|---|---|
| 传统生产链 | MINPA 原始/ori 数据 | 全能级 | H+、O+、O2+ | `day_spe`、`NV_MSO` | 历史成熟产品 |
| 全能级修订链 | `NV_MSO` + `day_spe` + MOMAG | 全能级 | H+、O+、O2+ | `NV_MSO_2`、MSE 网格归档和统计图 | 当前全能级生产口径 |
| 独立 Python 直算链 | 原始 `.2B`/ori + MOMAG | 全能级或严格 `E > 1 keV` | 全能级参考链为三物种；高能生产链为 O+、O2+ | 谱、密度、速度、标量温度（参考链）；高能日产品 | 公式审计与独立高能产品 |

共同的物理主线是：

```text
原始记录与 UTC
  → 按模式恢复能量、质量、角度维
  → 选择 H+ / O+ / O2+ 质量通道
  → 由 DPF 得到 DEF 能谱
  → 由能量和质量得到粒子速度
  → 以 dn = J·dE·dΩ/v 为权重计算零阶和一阶矩
  → 可选计算标量二阶矩
  → 仪器/星体坐标旋转到 MSO，再按需旋转到 MSE
  → 附加质量位、匹配状态、FOV 和来源元数据
  → 形成逐时刻产品与空间统计产品
```

## 2. 输入数据与来源追踪

### 2.1 MINPA 初始粒子数据

项目支持两类本地输入。

1. 公共 `.2B` 数据及其 XML/PDS 标签。标签提供固定宽度 UTC 字段、科学数据字段和记录布局；二进制载荷提供展平的 `Ion_Count`。
2. 已转换的 `ori/*.mat`。当前高能和质量标志流程优先使用 `D:\Data\TW-1\result\MINPA\ori`；只有在目标 UTC 日无重叠 ori 文件时，才回退到 `D:\Data\TW-1\rawdata\MINPA\public`。

原始目录保持只读。任何重排、拆分、质量标志、矩、坐标变换和空间统计都写入独立派生目录。

### 2.2 辅助几何数据

| 数据 | 主要变量 | 用途 | 当前时间匹配容差 |
|---|---|---|---|
| MOMAG `BssYYYYMMDD.mat` | `Roll_TW1`、`Pitch_TW1`、`Yaw_TW1` | 星体/探头姿态到 MSO | 5 s |
| MOMAG `BssYYYYMMDD.mat` | `P_TW1` | 航天器 MSO 位置；高能链还由差分求航天器速度 | 5 s |
| `R_MSO2MSE_Tianwen-1_YYYYMMDD.mat` | `R_MSO2MSE` | MSO 向量旋转到 MSE | 网格质量口径 8 s |

旋转矩阵使用前检查有限性、正交性和行列式接近 `+1`。每日 MSE 旋转时间轴先稳定排序；重复时刻优先保留有限元素最多的矩阵，同等完整时保持原始稳定顺序，并记录冲突数量和最大差异。

### 2.3 时间轴和跨日规则

- 原始记录按真实 UTC 过滤到目标日的 `[00:00:00, 次日 00:00:00)`，然后排序。
- 文件名日期只用于发现候选文件，不能替代记录内真实 UTC。
- `NV_MSO_2` 的密度时间 `NH_TW1(:,1)` 是三物种归档的权威逐行时间。有限的速度时间列只能用于一致性检查，不用于给 NaN 速度行重新配时。
- 相邻 `NV_YYYYMMDD.mat` 可能含重复时刻。全能级 MSE 归档只保留“记录真实 UTC 日期等于文件名日期”的行；该规则为每个唯一时刻选择正确日期的姿态来源。

## 3. 模式、记录拆分和数据立方体

### 3.1 模式识别

模式由文件名中的 `MINPA-MODn` 识别。当前粒子统计排除 mode 13。不同模式具有不同的能量中心、质量通道数和极角采样，因此每条记录必须携带 mode；不能假定整天使用同一能量表。

### 3.2 mode 12 特殊拆分

当 mode 12 的一条原始记录长度等于两个标准子记录时，按现有算法拆成两条子记录，时间置于原约 4.1 s 积分区间的四分之一和四分之三位置：

```text
t1 = t0 + 4.1/4 s
t2 = t0 + 3×4.1/4 s
```

当前质量标志生产实现确认，每个拆分子记录为 `32 × 48 × 1 = 1536` 个值。这个“1 angle”是当前 ori 产品的实际布局规则。普通模式的 16 方位角几何不能直接套到该子记录。

### 3.3 普通模式的数据立方体

对于通过形状检查的普通记录，展平向量恢复为：

```text
J[i, a, g] = differential particle flux
i: energy index
a: angular-bin index
g: mass-channel index
```

期望长度必须满足：

```text
Nvalue = Nenergy × Nangle × Nmass
```

长度不匹配时不得截断或猜测维度；该记录进入异常状态，谱或矩填 NaN，并保留来源与状态码。

## 4. 能量、质量和角度几何

### 4.1 能量表

各模式使用项目中固化的 MINPA 能量中心表，单位 eV。大多数表近似对数等距，但能量个数和比值随模式变化。混合模式输出必须保存逐记录的 `energy_eV_by_record`，不足最大长度的位置用 NaN 填充。

能量边界存在两种用途不同的模型：

- 独立直算参考产品的谱元数据：沿用 `dE/E = 0.15`，用 `E_low/high = E × (1 ∓ 0.075)` 描述中心附近的 15% 色散。
- `NV_MSO_2` 密度修订：用相邻对数能量中心比 `r = E(i+1)/E(i)` 推导完整能量箱宽度 `dE/E = sqrt(r) - 1/sqrt(r)`，然后取该模式的稳健代表值。

### 4.2 质量通道和物种选择

普通模式优先选择质量中心精确等于目标质量的通道：H+ 为 `1 amu`，O+ 为 `16 amu`，O2+ 为 `32 amu`。

| 模式 | 质量表概况 | 可直接形成的目标物种 |
|---|---|---|
| 1–6 | 8 个质量通道：1、2、4、16、38、32、44、65 amu | H+、O+、O2+ |
| 7–8 | 16 个质量通道，含 1、16、32 amu | H+、O+、O2+ |
| 9–11 | 单质量通道 1 amu | 仅 H+；O+/O2+ 无物种几何 |
| 12 | 32 个宽质量通道 | 按 Python 零基索引：H+ 用 0–1，O+ 用 18–19，O2+ 用 26–27 |

mode 12 多质量通道的能谱先分别计算，再在选中通道间做算术平均；矩积分则把各选中质量通道的 `dn` 单元合并求和。

### 4.3 普通模式角度和固体角

- 方位角覆盖 `0–360°`，16 个等宽箱，`dphi = 22.5°`。
- modes 1–6 的极角边界为 `0–90°` 的 4 个箱，因此普通几何有 `4 × 16 = 64` 个角度箱。
- modes 7–11 的极角边界为 `0–90°` 的 16 个箱，因此有 `16 × 16 = 256` 个角度箱。
- MINPA 只覆盖一个半球量级的视场，所有矩都可能受未观测半球影响。

对极角边界 `theta_j, theta_{j+1}` 和方位宽度 `dphi`，当前实现的固体角为：

```text
dOmega_j = [cos(theta_j + 90°) - cos(theta_{j+1} + 90°)] × dphi_rad
```

该固体角对 16 个方位角重复。实现中要求 `dOmega` 有限且为正，并检查总固体角处于仪器半视场的合理量级。

## 5. 从 DPF 得到三物种能谱

### 5.1 DPF 与 DEF

项目把 `Ion_Count` 解释为微分粒子通量：

```text
J(E,Omega,m) [cm^-2 s^-1 sr^-1 eV^-1]
```

对某个能量中心，微分能通量 DEF 为：

```text
DEF(E,Omega,m) = E × J(E,Omega,m)
```

当前文件中的谱单位字符串为 `eV/(s cm^2 sr eV)`，代数上等价于 `cm^-2 s^-1 sr^-1`。保留原字符串有利于说明它是由每 eV 的 DPF 乘能量得到的 DEF。

### 5.2 固体角加权的一维物种谱

对物种 `s` 的一个质量通道 `g`，一维 DEF 谱定义为：

```text
DEF_s,g(E_i) = E_i × [sum_a J(E_i,a,g) dOmega_a] / [sum_a dOmega_a]
```

若物种由多个质量通道组成，则：

```text
DEF_s(E_i) = mean_g(DEF_s,g(E_i))
```

这是一条“固体角加权平均 DEF 谱”，不是对固体角积分后的全向总能通量。绘图时仅对正且有限的 DEF 使用对数色标；时间轴为 UTC，能量轴为对数 eV。

### 5.3 谱产品的记录级元数据

每条谱记录至少需要以下伴随信息：真实 UTC、mode、有效能量箱数、逐记录能量中心、能量低/高边界、物种质量通道定义、谱单位、原始文件路径、质量标志和 mode 12 拆分状态。不同 mode 的谱不可只靠统一二维数组解释；必须同时读取逐记录能量表和 NaN 填充长度。

## 6. 速度单元和矩积分权重

### 6.1 由能量得到粒子速率

对能量 `E_i` 和质量 `m_g`：

```text
v_i,g = sqrt(2 e E_i / (m_g m_p))
```

其中 `e` 为元电荷，`m_p` 为质子质量，`m_g` 以 amu/质子质量倍数表示。计算后转换为 km/s；密度积分分母使用 cm/s。

### 6.2 速度方向

普通模式的每个角度箱先由极角和方位角构造单位视线方向，再应用固定的 MINPA 仪器到航天器星体坐标安装矩阵：

```text
C_body_from_MINPA = [[ 0, -1,  0],
                     [ 0,  0,  1],
                     [-1,  0,  0]]
```

等价分量映射为：

```text
[Xb, Yb, Zb] = [-V2, +V3, -V1]
```

该矩阵是正交且行列式为 `+1` 的固定旋转。旧 MATLAB 文本中曾出现另一种仅保速率但不保 MSO 分量的写法；当前项目以对本地 `NV/NV_MSO` 的数值回归确认上述矩阵为准。

### 6.3 单元密度权重

粒子强度与数密度的离散关系为：

```text
dn_i,a,g = J_i,a,g × dE_i × dOmega_a / v_i,g
         = DEF_i,a,g × (dE/E)_i × dOmega_a / v_i,g
```

单位检查：

```text
(cm^-2 s^-1 sr^-1 eV^-1) × eV × sr / (cm s^-1) = cm^-3
```

这项单位闭合是区分 DPF 与 DEF 的关键。如果把输入误当成 DEF 后再乘一次能量，密度会产生能量因子错误。

## 7. 各阶矩的定义和当前实现状态

### 7.1 零阶矩：数密度

```text
n_s = sum_i,a,g dn_i,a,g       [cm^-3]
```

仅对目标物种质量通道和所选能量范围积分。全能级产品使用该模式全部有效能量箱；高能产品使用严格条件 `E > 1000 eV`，不是 `E >= 1000 eV`，且不做航天器电势修正。

### 7.2 一阶矩：体速度

每个相空间单元的矢量速度为：

```text
v_vec_i,a,g = v_i,g × uhat_a
```

体速度为密度加权平均：

```text
U_s = [sum_i,a,g dn_i,a,g × v_vec_i,a,g] / n_s       [km/s]
```

如果密度有效但姿态缺失，仍可保留密度，而 MSO/MSE 速度置 NaN，并使用状态码标明“只有密度”。传统 `NV_MSO/NV_MSO_2` 速度不额外加航天器平动速度；独立 `>1 keV` Python 生产链默认由 `P_TW1` 有限差分求航天器 MSO 速度，并加到粒子速度上。比较两条支路时必须检查这一元数据差异。

### 7.3 二阶矩：标量温度

独立 Python 直算参考链定义相对体速度 `c = v_vec - U_s`，计算三维标量温度：

```text
T_s = [sum_i,a,g dn_m3 × m_g,kg × |c_m/s|^2]
      / [3 × n_m3 × e]                                      [eV]
```

它等于压力张量迹对应的各向同性标量温度。当前 `NV_MSO_2` 和全能级 HDF5 网格不保存此字段；因此不能从这些文件声称获得温度。

### 7.4 压力张量、各向异性和三阶矩

完整二阶中心矩可写为：

```text
P_s = sum_i,a,g m_g,kg × dn_i,a,g,m^-3 × c_i,a,g,m/s c_i,a,g,m/s^T    [Pa]
```

三阶热流可写为：

```text
q_s = 1/2 × sum_i,a,g m_g,kg × dn_i,a,g,m^-3 × |c_i,a,g,m/s|^2 × c_i,a,g,m/s
```

当前成熟生产链没有输出 `Pxx...Pzz`、平行/垂直温度、温度各向异性或热流。受 MINPA 半视场覆盖和 mode 12 角度布局影响，在实现这些高阶矩前必须先完成角度覆盖修正、误差传播和已知事件验证；本文公式只给出定义，不代表已有生产产品。

## 8. 能量宽度的两种当前口径

### 8.1 传统/独立直算口径

旧生产代码和独立直算参考实现使用固定：

```text
(dE/E)_old = 0.15
```

这使 `dn = DEF × 0.15 × dOmega / v`。

### 8.2 `NV_MSO_2` 的模式修订

`NV_MSO_2` 不重算每个能量箱，而是按模式整体缩放旧密度：

```text
r = E(i+1)/E(i)
(dE/E)_mode = median_i[sqrt(r) - 1/sqrt(r)]
n_new = n_old × (dE/E)_mode / 0.15
```

| 模式 | `(dE/E)_mode` | 密度缩放因子 |
|---|---:|---:|
| 1, 2 | 0.2339916 | 1.5599439 |
| 3, 4, 5, 7, 8 | 0.1446493 | 0.9643286 |
| 6 | 0.1940261 | 1.2935071 |
| 9, 10, 11 | 0.0998339 | 0.6655595 |
| 12 | 0.1509958 | 1.0066384 |

模式由 `day_spe` 的真实时间以 1 ms 容差匹配到 `NV_MSO` 行。三物种密度使用同一模式缩放因子；MSO 速度原样复制。因而 `NV_MSO_2` 的准确描述是“模式宽度校正后的旧矩产品”，不是“从原始 DPF 重新逐箱积分的产品”。

## 9. 从仪器坐标到 MSO 和 MSE

### 9.1 星体坐标到 MSO

使用最近邻 MOMAG 姿态：

```text
R_MSO_from_body = Rz(-Yaw) × Ry(-Pitch) × Rx(-Roll)
v_MSO = R_MSO_from_body × v_body
Xb_MSO = R_MSO_from_body × [1,0,0]^T
```

姿态时间差必须小于当前 5 s 容差。无匹配时保留密度和原始时间，速度与 `Xb_MSO` 填 NaN，不做外推。

### 9.2 MSO 到 MSE

对位置、速度和星体轴使用同一时刻附近的旋转：

```text
r_MSE  = R_MSO2MSE × r_MSO
v_MSE  = R_MSO2MSE × v_MSO
Xb_MSE = R_MSO2MSE × Xb_MSO
```

旋转保持矢量模长，因此 `|v_MSO|` 与 `|v_MSE|` 的差应只在浮点误差范围内。网格归档使用真实 UTC 日的 `P_TW1` 和旋转文件，不使用源 NV 文件名日期去猜几何。

### 9.3 FOV 与 `+Xb_MSE` 分类

MINPA 近似 `360° × 90°` 半球视场。`+Z_MSE` 是否在视场内，通过把目标方向投到探头/星体几何并与 mode 对应极角边界比较得到；姿态或 `Xb_MSO` 无效时标志为 NaN。

当前分类 X–Z 图另以 `Xb_MSE` 与 `±Z_MSE` 的夹角划分包含边界的 45° 圆锥：

```text
plus-Z class : angle(+Xb_MSE, +Z_MSE) <= 45°
minus-Z class: angle(+Xb_MSE, -Z_MSE) <= 45°
```

两类互斥，圆锥外记录只进入覆盖计数，不进入两类均值。当前指定 Y 薄层为 `-0.5 <= Y_MSE <= 0.5 Rm`。

## 10. 质量标志、有效性和状态码

### 10.1 五位物种独立质量掩码

H+、O+、O2+ 分别计算 uint32 五位问题掩码。

| 位 | 数值 | 判据 | 默认处理 |
|---|---:|---|---|
| bit 1 | 1 | 有效二维通道数为 5–10 | 警告但保留 |
| bit 2 | 2 | 有效二维通道数小于 5 | 剔除 |
| bit 3 | 4 | 超过 80% 的一维能量通道 `DEF > 1e5` | 剔除 |
| bit 4 | 8 | 七能段峰谷呈奇偶采集错误 | 剔除 |
| bit 5 | 16 | 任一二维通道 `DEF > 1e10` | 剔除 |

二维有效通道的原始 DPF 条件为 `0 < DPF < 1e8`。bit 4 在全能量、0–30、30–300、300–最大、0–50、50–500、500–5000 eV 七个区间检查相邻峰谷及奇偶相位。当前算法版本为 `2026-07-13-seven-band-all-species-v2-mode12-product-time`。

质量掩码必须同时读取 `quality_flag_available` 和时间匹配状态。`available=false` 时数值零只是缺测占位，不能解释成“无问题”。

### 10.2 矩有效性和兼容状态

全能级 59 字段网格为每个物种派生以下兼容状态：

```text
1 = 密度为正且有限，三个 MSE 速度分量也都有限
5 = 密度为正且有限，但速度不完整
4 = 密度非有限或非正（包含零信号）
```

源密度原值仍保存；状态不是对源数据的替代。独立高能链还有“无几何/物种高能单元”“意外形状”等额外处理状态，解释时应读取对应文件元数据，而不是套用全能级三状态表。

## 11. 最终产品怎样形成

### 11.1 `day_spe`：逐记录能谱

传统 MINPA 处理按 mode、物种和时间保存固体角加权的一维 DEF 谱。它也是 `NV_MSO_2` 匹配 mode 的时间来源。混合模式时，谱值必须与该记录的能量中心表一起读取。

### 11.2 `NV_MSO`：全能级旧矩

旧链由全能量范围计算 H+、O+、O2+ 密度和体速度，再用 MOMAG 姿态旋转到 MSO。重复时间和无姿态行按旧逻辑保留 NaN/去重结果。该产品使用固定 `dE/E = 0.15`。

### 11.3 `NV_MSO_2`：当前全能级矩源

输出变量为：

```text
NH_TW1, NO_TW1, NO2_TW1                     [time, n] cm^-3
VH_TW1_MSO, VO_TW1_MSO, VO2_TW1_MSO        [time, Vx, Vy, Vz] km/s
Xb_MSO                                      [time, x, y, z] unit vector
mode_TW1                                    [time, mode]
density_scale_TW1                           [time, scale]
energy_width_over_E_TW1                     [time, dE/E]
NV_MSO_2_info                               provenance metadata
```

完整构建含 1051 个日产品和 6,549,739 个物理行；由于相邻文件跨日重复，唯一时刻为 5,548,505。

### 11.4 全能级 MSE 网格归档

- 来源：`NV_MSO_2`，不重新积分原始谱。
- 物种：H+、O+、O2+。
- 坐标：MSE。
- 网格：X/Y/Z 均为 `[-5,+5] Rm`，分辨率 `0.1 Rm`，`Rm=3397 km`，`+5 Rm` 进入最后一格。
- 组织：每个非空三维格点一个 HDF5 文件，内部按真实 UTC 日期保存逐记录 59 字段复合数据，不跨时刻求和或平均。
- 正式结果：1029 个有几何日期，22 个缺 MOMAG 与旋转日期排除；3,979,338 条记录落入网格。

### 11.5 X–Z 统计图

有效记录先按物种状态和质量位筛选。每条记录先计算速率 `sqrt(Vx^2+Vy^2+Vz^2)`，再在 X–Z 格内以记录数加权求密度均值和速率均值。密度—速率通量定义为：

```text
flux = mean_density_cm3 × mean_speed_km_s × 1e5    [cm^-2 s^-1]
```

它是两个均值的乘积，不等同于逐记录 `n×v` 的均值。重分箱时必须累加原始计数和物理量和，再重算均值；不能直接平均各格均值。

## 12. 可复现伪代码

```python
for raw_record in discover_minpa_records(day):
    t, mode, flat_J = decode_and_validate(raw_record)
    for t_sub, flat_sub in split_mode12_if_required(t, mode, flat_J):
        energy, mass, angle, dOmega = geometry_for(mode, product_layout)
        cube = reshape_only_if_exact(flat_sub, energy, angle, mass)

        for species in [Hplus, Oplus, O2plus]:
            groups = species_mass_groups(mode, species)
            spectrum_DEF = mean_over_groups(
                energy * sum_over_angle(cube[:, :, groups] * dOmega)
                / sum(dOmega)
            )

            for each selected energy, angle, mass cell:
                speed = sqrt(2 * e * energy / mass)
                dn = J * dE * dOmega / speed
                v_body = speed * look_direction

            density = sum(dn)
            velocity_body = sum(dn * v_body) / density
            temperature = sum(dn * mass * abs(v_body - velocity_body)**2) \
                          / (3 * density * e)

        R_mso_body = Rz(-yaw) @ Ry(-pitch) @ Rx(-roll)
        velocity_mso = R_mso_body @ C_body_minpa @ velocity_minpa
        write_record_with_mode_energy_quality_and_provenance()
```

对于当前 `NV_MSO_2`，伪代码中的“逐箱重新积分”由以下派生步骤替代：

```python
mode = nearest_day_spe_mode(epoch, tolerance=0.001)
density_new = density_NV_MSO * mode_dE_over_E[mode] / 0.15
velocity_mso_new = velocity_mso_NV_MSO
Xb_MSO = R_mso_body @ [1, 0, 0]
```

## 13. 强制科学检查

1. 形状：每条普通记录严格满足 `Nenergy × Nangle × Nmass`；mode 12 使用其产品路径的实际拆分布局。
2. 单位：`J×dE×dOmega/v` 必须闭合为 `cm^-3`；速度统一为 km/s；温度为 eV。
3. 物种：mode 9–11 不应产生 O+ 或 O2+ 有效矩。
4. 能量：高能选择严格为 `E > 1000 eV`，全能级不应用该筛选。
5. 坐标：固定安装矩阵和姿态矩阵均为 proper rotation；MSO→MSE 前后速率保持。
6. 时间：逐行真实 UTC 与产品日、mode、姿态、位置和旋转的时间差均在各自容差内。
7. 密度量级：只检查正值和合理数量级，不以单一阈值替代质量位；极低密度可标记“不可靠”但不应静默删除源值。
8. FOV：报告半视场限制，不能把未观测半球当作零通量。
9. 统计语义：跨记录报告均值/中位数；记录和只作为审计量。
10. 产品闭合：源行数、唯一时刻数、排除日期、网格内记录数、日汇总和 HDF5 行数必须完全一致。

## 14. 已知限制与后续工作

- `Ion_Count` 的 DPF 解释是当前代码和量纲审计的项目假设；正式发表前仍应在官方产品标签/定标文档中逐变量核对名称、单位、填充值和有效范围。
- `NV_MSO_2` 只对旧密度做模式整体比例修订，没有逐能量箱使用各自边界重积分；若要求最高精度，应从原始 DPF 直接用逐箱 `dE_i` 重算并与现产品回归。
- mode 12 的实际 ori 布局与普通 16 方位角几何不同。独立直算参考脚本中的通用 mode 12 角度假设在适配完成前不能作为该模式的生产矩结果。
- 标量温度只存在于独立参考链；压力张量、各向异性和热流尚未形成经过 FOV 校正与验证的生产产品。
- 半视场导致密度和矢量矩为“观测覆盖内的矩”。对于强各向异性分布，偏差可能比数值积分误差更重要。
- 传统全能级速度不加航天器平动速度，而独立高能 Python 链默认添加；跨产品比较前必须统一参考系定义。
- 全能级网格的 `Rm=3397 km` 是该产品的固定元数据，不应从通用常量模块自动替换。

## 15. 源代码与方法文件索引

| 文件 | 作用 |
|---|---|
| `src/highE/tw1_minpa.py` | 原始 MINPA 读取、普通模式几何、严格 `E>1 keV` O+/O2+ 密度和速度、MSO/MSE 旋转 |
| `scripts/build_tw1_minpa_nv_mso_2.m` | 由 `NV_MSO` 生成模式宽度修订后的 `NV_MSO_2` |
| `scripts/tw1_minpa_quality_flag_all_species_core.m` | 三物种五位质量标志核心算法和 mode 12 特殊布局 |
| `scripts/tw1_minpa_quality_flag_config.m` | 质量阈值、七能段和版本配置 |
| `src/highE/tw1_all_energy_mse_grid_records.py` | 全能级三物种逐格 HDF5 归档、真实日期去重和 MSE 旋转 |
| `src/highE/tw1_mse_xz_stats.py` | 全能级三物种 X–Z 记录加权统计（入口 `run_tw1_all_energy_mse_xz_statistics`） |
| `src/highE/tw1_all_energy_mse_xz_xb_stats.py` | 按 `+Xb_MSE` 与 `±Z_MSE` 关系分类统计 |
| `docs/method_tw1_minpa_instrument_to_orbiter.md` | 固定安装矩阵验证 |
| `docs/method_tw1_minpa_quality_flag.md` | 五位质量位定义和审计 |
| `docs/method_tw1_minpa_nv_mso_2.md` | 全能级密度宽度修订和输出字段 |
| `docs/method_tw1_all_energy_mse_grid_records.md` | 59 字段 MSE 网格归档和最终审计 |
| `docs/minpa_density_dpf_audit.md` | DPF/DEF 与密度公式量纲审计 |
| `D:\Code\TW-1\TW1_MINPA_deal_v5.m` | 传统谱和矩的历史实现依据 |
| `D:\Code\TW-1\TW1_MINPA_redeal_3.m` | 传统去重与姿态旋转实现依据 |
| `D:\codex\处理MAVEN数据\scripts\tw1_minpa_moments.py` | 原始数据到三物种谱、密度、速度、标量温度的独立参考实现 |

## 16. 变量速查

| 符号/变量 | 含义 | 单位 |
|---|---|---|
| `J` / `Ion_Count` | 微分粒子通量 DPF | `cm^-2 s^-1 sr^-1 eV^-1` |
| `DEF` | `E×J`，微分能通量 | `eV/(s cm^2 sr eV)` |
| `E_i` | 第 i 个能量中心 | eV |
| `dE_i` | 能量箱宽度 | eV |
| `dOmega_a` | 第 a 个角度箱固体角 | sr |
| `m_g` | 第 g 个质量通道 | amu 或 kg |
| `dn` | 单个能量—角度—质量单元的密度贡献 | `cm^-3` |
| `n_s` | 物种数密度，零阶矩 | `cm^-3` |
| `U_s` | 物种体速度，一阶矩 | `km/s` |
| `T_s` | 三维标量温度，二阶中心矩的迹 | eV |
| `P_s` | 压力张量，当前未生产 | Pa |
| `q_s` | 热流，当前未生产 | `W/m^2` |
| `R_MSO_from_body` | 星体坐标到 MSO 的姿态旋转 | 无量纲 |
| `R_MSO2MSE` | MSO 到 MSE 的旋转 | 无量纲 |

---

文档结论：当前可审计地得到的最终量包括三物种固体角加权 DEF 能谱、全能级密度和 MSO/MSE 三分量体速度；独立直算参考链还给出标量温度。完整压力张量和三阶热流尚不是当前生产产品。所有结果必须与 mode、逐记录能量表、FOV、质量位、时间匹配和坐标元数据一起解释。
