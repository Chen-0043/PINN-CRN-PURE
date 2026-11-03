# PINN 驱动全基元 CRN 数字孪生研究

本代码包用于复现deGFP单蛋白及Pgi–Eno六蛋白体系的教师模型模拟、投料扰动分析、PINN先验轨迹生成、物理约束轨迹校正和相关结果评价。六蛋白教师模型包括Pgi、PfkA、TpiA、GAPDH、Pgk、Eno六个单表达体系，以及一个六蛋白共表达体系；六蛋白PINN模拟针对共表达体系。

## 1. 结构

| 子文件夹 | 任务 | 主要结果 |
|---|---|---|
| [deGFP 合成体系教师模型间歇式模拟](<deGFP 合成体系教师模型间歇式模拟/>) | 根据包内模型、初始条件和参数重新积分deGFP间歇式体系，与文献浓度数据比较 | 间歇式浓度轨迹、0–6 h文献比较误差；积分输出覆盖8 h |
| [deGFP 合成体系教师模型连续模拟](<deGFP 合成体系教师模型连续模拟/>) | 模拟连续体系至72 h，评价48–72 h的稳态偏差，并导出0–6 h全状态参考数据 | 连续浓度轨迹、稳态指标和PINN教师监督数据 |
| [Pgi-Eno 6 蛋白合成体系教师模型间歇式模拟](<Pgi-Eno 6 蛋白合成体系教师模型间歇式模拟/>) | 分别模拟六个单表达体系及一个共表达体系，比较3 h、6 h文献参考值 | 七个体系的浓度结果及14个比较点的误差 |
| [Pgi-Eno 6 蛋白合成体系教师模型连续模拟](<Pgi-Eno 6 蛋白合成体系教师模型连续模拟/>) | 分别模拟七个体系至72 h，另生成共表达体系0–6 h的PINN监督数据 | 七组长时程轨迹、稳态指标及9198个可比对象的参考轨迹 |
| [教师模型扰动模拟](<教师模型扰动模拟/>) | 计算五类投料的6 h deGFP终点、容差收敛性、参数联合扰动及相邻投料点；重算拓扑指标 | 完整剂量响应、最优投料倍数、数值收敛和局部稳定性结果 |
| [deGFP 合成体系 PINN 先验轨迹生成](<deGFP 合成体系 PINN 先验轨迹生成/>) | 加载正式已训练权重，重新生成全状态先验，评价3151个可比代谢物 | deGFP第一阶段轨迹、MRE、R²和NRMSR |
| [Pgi-Eno 6 蛋白合成体系先验轨迹生成](<Pgi-Eno 6 蛋白合成体系先验轨迹生成/>) | 加载正式已训练权重，重新生成共表达全状态先验，评价9198个可比对象 | 六蛋白第一阶段轨迹、MRE、R²和NRMSR |
| [deGFP 合成体系 PINN 物理约束轨迹校正](<deGFP 合成体系 PINN 物理约束轨迹校正/>) | 从本目录的已训练权重生成先验，再执行三段各1000步的节点校正 | deGFP第二阶段轨迹、浓度误差、轨迹和模块指标 |
| [Pgi-Eno 6 蛋白合成体系 PINN 物理约束轨迹校正](<Pgi-Eno 6 蛋白合成体系 PINN 物理约束轨迹校正/>) | 从已训练权重生成先验，依次执行完整的校正 | 六蛋白第二阶段轨迹、浓度误差、轨迹和模块指标 |

九个工作流均携带自身需要的研究数据、模型、参数及参考结果，可以单独复制运行。目录之间保留必要数据副本，是为了保证各工作流的独立性。

## 2. 模拟流程与复现起点

研究流程为：

```text
包内模型、序列、参数和初始条件
                  ↓
间歇式教师模型模拟与文献比较
                  ↓
连续式教师模型模拟、浓度参考轨迹与投料扰动分析
                  ↓
第一阶段：神经网络生成全状态先验轨迹
                  ↓
第二阶段：逐段优化离散浓度节点
                  ↓
最终浓度轨迹 → 误差与机制一致性评价 → 图表
```

默认PINN“复现”使用包内已经训练好的神经网络权重，重新进行前向计算；校正阶段随后从生成的先验开始执行完整节点优化。各校正段以上一段的输出为输入，最终指标从最终轨迹重新计算。


## 3. 运行环境

运行脚本使用Bash，并调用当前环境中的`python3`。请先激活对应Python环境，再运行脚本。

| 工作流 | 已验证的Python环境 | 主要依赖 |
|---|---|---|
| 四个教师模型目录和教师模型扰动模拟 | CPython 3.14 | NumPy、pandas、SciPy、Matplotlib、Tellurium/RoadRunner、python-libsbml |
| 四个PINN目录 | Python 3.11 | PyTorch、NumPy、pandas、SciPy、Matplotlib |

每个子目录的`requirements.txt`列出本工作流依赖的版本；`verified_environment.json`记录已有验证环境。六蛋白教师流程需要包内`legacy_runtime/`中的CPython 3.14字节码，这些文件属于必要计算依赖，应保留。

以下以在总目录中创建两个独立软件环境为例。环境放在代码包外层，不增加工作流子目录。

```bash
# 当前目录：本README所在的总目录

# 教师模型环境
python3.14 -m venv ../pure-teacher-env
source ../pure-teacher-env/bin/activate
python -m pip install -r "deGFP 合成体系教师模型间歇式模拟/requirements.txt"
# 运行其他教师/扰动目录前，也按对应目录的requirements.txt安装依赖
deactivate

# PINN环境
python3.11 -m venv ../pure-pinn-env
source ../pure-pinn-env/bin/activate
python -m pip install -r "deGFP 合成体系 PINN 先验轨迹生成/requirements.txt"
python -m pip install -r "Pgi-Eno 6 蛋白合成体系先验轨迹生成/requirements.txt"
```

如果已有符合要求的环境，可直接激活使用。模型输入和研究数据均由包内提供；Python及第三方软件库需要在运行电脑上安装。

## 4. 通用运行方式

进入需要运行的子文件夹，执行：

```bash
bash 运行.sh 检查
bash 运行.sh 复现
bash 运行.sh 评价
```

| 命令 | 用途 |
|---|---|
| `bash 运行.sh 检查` | 检查本目录必要资产或参考结果；六蛋白校正目录还会从参考数组重算主要指标 |
| `bash 运行.sh 复现` | 执行本目录正式模拟流程，并按入口配置评价新结果 |
| `bash 运行.sh 评价` | 单独计算本目录结果指标；评价脚本会打印实际读取的结果来源 |
| `bash 运行.sh 绘图` | 四个PINN目录及扰动目录支持，用于生成相应图形 |

例如，复现六蛋白完整物理约束校正：

```bash
# 在总目录中，先激活PINN环境
source ../pure-pinn-env/bin/activate
cd "Pgi-Eno 6 蛋白合成体系 PINN 物理约束轨迹校正"
bash 运行.sh 检查
bash 运行.sh 复现
bash 运行.sh 评价
bash 运行.sh 绘图
```

`复现`会执行实际计算。`检查`通过只表示相应检查通过，不能代替完整模拟。新结果通常写入`reproduced_results/`；正式参考结果保留在`reference_results/`或模型数据目录中。

### 附加命令

在对应子文件夹内运行：

| 目录 | 命令 | 用途 |
|---|---|---|
| 两个连续式教师目录 | `bash 运行.sh 监督数据` | 单独重新生成PINN教师参考数据 |
| 教师模型扰动模拟 | `bash 运行.sh 端点` | 重新计算五类投料的完整剂量终点 |
| 教师模型扰动模拟 | `bash 运行.sh 不确定性` | 运行容差收敛和参数联合扰动相关任务 |
| 教师模型扰动模拟 | `bash 运行.sh 局部最优` | 计算原最优点相邻的投料点 |
| 教师模型扰动模拟 | `bash 运行.sh 拓扑` | 从包内网络清单重算拓扑指标 |
| 两个PINN先验目录 | `bash 运行.sh 训练` | 按保存的训练流程从初始化训练；与默认加载已训练权重的复现方式不同 |
| deGFP PINN校正目录 | `python3 run_benchmark27_from_weights.py` | 运行27节点、1000步校正性能基准 |
| deGFP PINN校正目录 | `bash run_independent.sh performance verify` | 检查`性能对比/`中的性能比较材料 |

## 5. 文件与数据的用途

不同工作流保留原有文件结构，以下名称按实际出现的目录使用：

| 文件或目录 | 用途 |
|---|---|
| 总目录`README.md` | 本代码包唯一的说明文档，集中记录九个工作流的任务、运行方式和结果 |
| `运行.sh` | 推荐使用的统一中文运行入口 |
| `run_independent.sh` | 对应底层任务的运行入口；命令分支见脚本 |
| `requirements.txt`、`verified_environment.json` | 依赖版本及已验证软件环境记录 |
| `evaluate_results.py` | 根据浓度数组、教师参考或扰动数据重新计算指标 |
| `inputs/`、`model_inputs/` | 初始投料、补料、表达对象配置和阶段输入等数据 |
| `model_assets/`、`PURE_CRN_models/`、`external_PURE_CRN_models_sparse/` | 网络、序列、稀疏矩阵、状态映射、动力学参数等模型资产 |
| `src/`及子目录内Python脚本 | 模型构建、模拟、训练、校正和评价实现 |
| `legacy_runtime/` | 六蛋白教师模型所需的历史运行实现与字节码 |
| `historical_runtime/` | 六蛋白校正各段所需的实现及`correction_recipe.json`阶段配置 |
| `prior_initialization_recipe.json` | 先验模型的训练阶段配置 |
| `prior_weight_replay_recipe.json` | deGFP校正目录从保存权重生成先验的配置 |
| `formal_result_spec.json` | 当前六蛋白校正结果的版本、权重哈希和预期数值 |
| `reference_results/` | 正式参考结果，用于比较或在尚未重算时进行评价 |
| `results/`、`run_outputs/` | 原有结果数据或指定任务的计算输出；以对应脚本打印的来源为准 |
| `reproduced_results/` | 运行后生成的新轨迹、指标、阶段记录等；部分其他输出位置见下一节 |
| `raw_literature_data/`、六蛋白文献比较CSV | 文献参考数据及包内整理的比较数据 |
| `MODEL_CONFIGURATION.json`、`PARAMETER_MANIFEST.csv` | 相应教师模型的机器可读配置和参数清单 |
| `FILE_MANIFEST_SHA256.txt`及完整性校验脚本 | 模型输入和代码的哈希校验数据；属于机器读取的校验文件 |
| `model_evaluation_20260730/` | deGFP PINN的网络资产、正式权重、轨迹、约束数据和计算脚本 |
| `绘图/` | 绘图脚本、SVG模板、字形轮廓和图形输出 |
| `性能对比/` | deGFP传统ODE与PINN性能比较的脚本及记录 |

`.pt`文件保存已训练神经网络参数；`.npz`、`.npy`保存浓度、时间或数值矩阵；XML及相关文本保存模型；CSV、JSON保存数据、配置和评价结果。各目录内的相对链接指向本工作流中的资产，复制目录时应保持结构和链接有效。

## 6. 主要输出位置

下表路径均相对于对应工作流子目录。

| 工作流 | 主要模拟输出 | 主要评价输出 |
|---|---|---|
| deGFP间歇式教师 | `reproduced_results/batch/batch_timeseries.csv` | `reproduced_results/current_metrics.json` |
| deGFP连续式教师 | `reproduced_results/continuous/gamma_0p5_continuous_72h_timeseries.csv`；全状态参考由`export_all_species_reference.py`另行导出 | `reproduced_results/current_metrics.json` |
| 六蛋白间歇式教师 | `reproduced_results/seven_systems/` | `reproduced_results/current_metrics.json`，包含七个体系、14个比较点 |
| 六蛋白连续式教师 | `reproduced_results_verified/single_<蛋白名>/timeseries.csv`和`reproduced_results_verified/coexpression/timeseries.csv`；监督数据在`reproduced_supervision_source/` | `reproduced_results/current_metrics.json` |
| 教师模型扰动 | `run_outputs/`及相应任务输出目录 | `reproduced_results/current_metrics.json`、`reproduced_results/perturbation_conclusion_audit.json` |
| deGFP PINN先验 | `model_evaluation_20260730/article_gapfill_20260730/full_elementary_pinn/replayed_frozen_formal_prior_prediction.npz` | `reproduced_results/current_metrics.json` |
| 六蛋白PINN先验 | `reproduced_results/replayed_prior/prior_prediction_6h.npz` | `reproduced_results/current_metrics.json` |
| deGFP PINN校正 | `model_evaluation_20260730/article_gapfill_20260730/full_elementary_pinn/from_prior_stage_3_shooting_nodes.npz` | `reproduced_results/current_metrics.json` |
| 六蛋白PINN校正 | `reproduced_results/full_history/`保存逐段结果；最终轨迹在`reproduced_results/final_correction_5p5/corrected_prediction_6h.npz` | `reproduced_results/current_metrics.json`、`reproduced_results/four_reported_losses.json`及物理评价明细 |

PINN绘图输出位于`绘图/`，当前图形文件为`图4_PINN先验模拟结果_统一0p24.svg`和`图5_PINN物理校正结果_统一0p24.svg`。字形轮廓随包保存，用于维持已确认的字体、字号和间距。六蛋白校正目录的图5数据会从本目录浓度数组重新提取；其余目录的共用图形部分采用包内冻结数据和模板。

## 7. 当前主要模拟结果

### 教师模型

| 任务 | 当前参考结果 |
|---|---|
| deGFP间歇式文献浓度比较 | 0–6 h MRE为10.85% |
| 六蛋白间歇式文献比较 | 六个单表达体系及一个共表达体系，共14个比较点，MRE为18.88% |
| deGFP连续式稳态 | 48–72 h浓度相对该区间均值的最大绝对偏差约为1.5648×10⁻³ µM |
| 六蛋白连续式稳态 | 七个体系48–72 h的最大绝对偏差约为9.4345×10⁻⁶～6.6072×10⁻⁵ µM；共表达体系约为1.2672×10⁻⁵ µM |

这里的稳态偏差定义为相对区间均值的最大偏差；它与区间最大值减最小值的“极差”不同。文献比较数据及换算口径保存在各教师模型目录内。

### 投料扰动

扰动倍数为0、0.01、0.03、0.06、0.1、0.22、0.25、0.5、1、2、3、4；五个对象共60个终点。当前模型和扫描网格下的6 h产量最优倍数为：

| 投料对象 | 最优倍数 |
|---|---:|
| ATP | 0.5 |
| GTP | 3 |
| CTP | 0.01 |
| UTP | 0.01 |
| 总氨酰化模块 | 0.22 |

数值收敛性和参数联合扰动结果由本目录相应入口及`verify_perturbation_conclusions.py`核对。这些最优点对应所使用的模型、边界条件和离散投料网格。

### PINN浓度与轨迹评价

MRE和R²在6 h节点计算。deGFP和六蛋白分别纳入3151、9198个可比对象；MRE分母使用`max(|教师浓度|, 10⁻¹² M)`。下表NRMSR为满足`s_i^rhs > 1 µM`条件的轨迹的逐轨迹算术平均。

| 体系与阶段 | 6 h MRE | 6 h R² | 满足阈值的轨迹数 | 平均NRMSR |
|---|---:|---:|---:|---:|
| deGFP先验 | 17.25% | 0.9788 | 91 | 105.90% |
| deGFP校正 | 23.57% | 0.9771 | 54 | 47.76% |
| 六蛋白共表达先验 | 0.02% | 0.9995 | 47 | 102.77% |
| 六蛋白共表达校正 | 0.92% | 0.9863 | 34 | 23.09% |

其他阈值及全状态指标见评价输出。表中的选取轨迹NRMSR不能替代全部轨迹或全部隐藏状态的NRMSR。deGFP校正后54条轨迹中，按当前分层统计剔除13条高误差轨迹，剩余41条平均NRMSR为30.04%。

目前正文采用的模块指标为：deGFP的IF1/IF3介导翻译起始装配模块28.11%、总能量再生模块54.66%；六蛋白的CTP和UTP转录底物掺入子模块分别为32.96%、35.19%。模块成员及统计方式由各评价脚本定义。

六蛋白校正轨迹上的四项论文损失为：

| 损失项 | 数值 |
|---|---:|
| `L_data` | 0.011423593740873065 |
| `L_int` | 0.05470841435882209 |
| `L_key` | 0.2188060416700384 |
| `L_prox` | 2.003624516982231×10⁻⁶ |
| `L_data + L_int + 2L_key + 0.005L_prox` | 0.5037441014578945 |

六蛋白实际校正仍保留原有辅助优化项；上表为最终轨迹的四项事后评价。完整阶段配置和实际损失组成见`historical_runtime/correction_recipe.json`、各阶段输出及校正脚本。

## 8. 模型来源、分析与绘图补充

### 模型数据和序列来源

deGFP教师模型的基础SBML、逐位转录来源、投料CSV、参数JSON和连续交换配置随各目录提供。原模型中逐核苷酸转录与逐氨基酸有效翻译是不同的分辨率；模型分类和参数记录见`results/reaction_kinetics_audit/`中的CSV及JSON。该目录保存`reaction_kinetics_inventory.csv`逐反应清单、`parameter_values.csv`参数表、`reaction_class_summary.csv`分类统计和`generated_model_counts.json`模型规模。机器可读清单保留完整数据。

deGFP教师模型实际包含3399个状态，3151是与精细CRN对应的可比代谢物数；间歇式、连续式反应数分别为3696和3762。deGFP精细CRN包含5772个状态和7061条基元反应。六蛋白共表达精细CRN包含49143个精细状态和57368条基元反应；当前物理算子另含6个辅助池，实际状态向量为49149维。相关状态清单、映射矩阵和模型参数以包内资产为准。

六蛋白文献参考为论文荧光信号经既定换算得到的浓度比较值；共表达的All信号是总量，不能作为六种蛋白各自浓度的直接实验测量。`inputs/fusion_batch_validation/`中的CSV保留融合构建来源和假设记录：原生酶编码序列对应MG1655参考序列，涉及的msfGFP标签长度为238 aa，另有SUMO标签敏感性方案。文献未提供的标签编码序列采用预定密码子规则重建，并非作者原始构建序列。正式模拟采用哪个模板及DNA投料配置，由运行入口实际指向的panel和投料文件确定；这些辅助来源材料不会自动替换正式输入。

### 扰动模拟、机制图及原始数值表

`bash 运行.sh 复现`计算完整投料扫描、容差收敛、参数扰动与相邻投料点；`bash 运行.sh 绘图`重建文章机制图。机制图部分使用包内已经整理的CSV等分析数据。图形重建与ODE重新积分是两项任务。

文章扰动热图由`run_article_3_3_figures.py`调用`technical_report_pure_20260713/`中的两个轴特异性热图脚本生成，输出在`figures/article_331_20260813/`。零投料图比较零倍与最优点；过量投料图采用以下比较：

| 对象 | 过量比较点／最优点 |
|---|---|
| ATP | 1／0.5 |
| GTP | 4／3 |
| CTP | 0.03／0.01 |
| UTP | 0.03／0.01 |
| 总氨酰化模块 | 0.25／0.22 |

已有机制时序数据的统一采样时刻为0.02、0.05、0.1、0.15、0.2、0.3、0.5、0.75、1、1.25、1.5、1.75、2、2.25、2.5、2.75、3、4、5、6 h。对应重跑入口为`run_latest_perturbation_simulations.py`，输出在`run_outputs/latest_ntp_unified_time`、`latest_total_aars_unified_time`及`latest_metrs_enzyme_only_unified_time`。MetRS专门分析保留在相关数据和脚本中，不属于上述五类正文扰动热图。

历史技术报告中的数值表继续以CSV等数据文件保存，结果图继续以SVG、PNG或PDF保存。旧Word技术报告及其专用Word图片替换脚本已移出正式包；正文模拟、数值评价及机制图生成不依赖这些Word文件。

机制图按“产物与副产物—模板和核糖体承接—翻译位置链与通量—能量及扰动对象”的顺序阅读。原变量名称和配色由绘图脚本保留；其中`tl_gtp_progression`与`translation_GTP_progression`均对应图中的翻译GTP推进指标。各原始CSV列名及其含义保留在数据表和脚本中。

### 性能比较记录的使用

性能材料在deGFP PINN校正目录的`性能对比/`中。1000步、27节点的校正基准与生成正式最终结果的完整校正链分别计算；单次基准耗时不能直接当作完整流程耗时。神经网络前向预测、教师数据生成、训练和轨迹校正的计时对象也应分别查看相应JSON记录。

其中`同算子核查_20261102/`另提供`prove_operator_identity.py`和`run_sameoperator_probe.py`。在该子目录、PINN环境中可以运行：

```bash
python3 prove_operator_identity.py
python3 run_sameoperator_probe.py BDF --timeout 180
python3 run_sameoperator_probe.py Radau --timeout 180
```

这些命令核查所用算子及ODE输出，结果写入该目录的`reproduced_results/`。原性能基准与后续同算子核查的设置及结果分别保存在各自JSON中，不能把不同设置的测量记录混作同一场比较。JSON中记录的历史绝对路径属于溯源字段；正式入口读取的是包内模型和参数。

### 运行故障定位

先检查激活的Python环境及其`requirements.txt`依赖，再执行对应工作流的`bash 运行.sh 检查`。六蛋白教师流程应使用CPython 3.14并保留`legacy_runtime/`；复制目录时保留相对链接及目录结构。若指标与上表不同，应先确认评价输出打印的轨迹来源、监督映射、单位、阈值和阶段配置，不改动保存权重或原始模型资产来迁就指标。
