最新纯机制 v4（2026-10-27 已完成）
结果：runs/pilot_seen_dose/experiment_v4_pure_mechanism_fixed/RESULTS.html。
仅药物机制＋对照＋code语义，四组信息消融，64 case-arms全部封存并独立复核。
主alpha=.3：full ΔRMSE=.356299，不编辑=.355724，未改善；去机制=.386305、去语义=.396323、两者均隐藏=.372037。
完整信息减少部分错误编辑，但尚未得到优于不编辑的药物预测。16孔/786处理细胞为已揭盲开发验证。
复现与限制见V4_README.txt。新的独立入口是pure_mechanism_experiment.py；原active_experiment.json仍指向v2。

以下为先前版本记录：

LEXA + 纯文字 DeepSeek agent：药物响应试验流水线

当前已完成的真实实验（2026-10-26）
工作目录 runs/pilot_seen_dose，实际活动结果目录为 experiment_v2_feedback，
由 active_experiment.json 指定。32轴/512状态已完成DeepSeek文字标注；
全部16个ready wells × one_shot/agent共32份预测先封存，再统一揭盲评分。
首轮agent ΔRMSE=0.349634，one_shot=0.349729，不编辑=0.355724，
同药物最近训练剂量=0.204749。当前文字编辑仅略好于不编辑，未超过简单基线。
范围：A549，4药物，留出5µM，固定24h，16wells/786处理细胞/2000选定基因。
最终报告位于 experiment_v2_feedback/report/RESULTS.html（或RESULTS.txt）。
v1 experiment/ 因重复轴工具输入导致执行中止，未评分，保留原始失败记录；
v2仅改善该输入错误的反馈，在读取test真值前统一重跑，不混合两轮预测。

当前工作目录继续运行必须保持全量case与原方法/预算：
bash run.sh --case-limit 0 --methods one_shot,agent --max-rounds 8 --api-key-file /cpfs_speech/user/zhoutianhong.zth/.secrets/deepseek.key

只重算已有预测的评分和报告，不调用API：
python pipeline.py --stage score --case-limit 0
python report_results.py --experiment runs/pilot_seen_dose/experiment_v2_feedback --fitted runs/pilot_seen_dose/lexa --output runs/pilot_seen_dose/experiment_v2_feedback/report --previous-experiment runs/pilot_seen_dose/experiment

先前4case示例是新work的默认screening配置；不要用它缩小已完成的16case计划。
后续按本轮结果调整prompt/表示或训练时，这批5µM数据应视为开发集；
最终泛化结论需要新的独立测试，不能重复调本留出集来声称有效。

真实扰动前后 code 与编解码趋势（2026-10-26 已完成）
runs/pilot_seen_dose/codec_changes/report/RESULTS.html 包含六张图：
32轴TV、32×16状态占用差、主要轴状态分布、全部剂量描述趋势、
真实vs重建基因变化散点、主要正负基因变化。
分析使用同一个冻结best.pt编码真实对照和处理细胞，无DeepSeek调用。
9028个唯一细胞；167有效注释处理孔中111个满足各组>=20cells，
56个排除（含9个QC后0cell孔），28药物剂量条件中25个有合格孔。
主结果仍优先看5µM的16孔/786处理cells；其他剂量包括训练/验证数据，
只用于描述。对照严格匹配drug+plate+replicate，孔等权平均，
无单细胞前后配对，不把code ID当连续数值。
cell_codes.npz保存cell×32 codes及原始/重建2000基因矩阵；
case/condition_state_occupancy.csv保存32×16占用率及差值。
case/condition_axis_changes.csv保存TV/JSD及轴级基因变化幅度。
condition_codec_metrics.csv分别给mean_vector_*和mean_well_*，不可混用。
manifest.json及独立verification.json记录来源、数学恒等式和范围核验。

重新计算请使用新目录，保留已核验产物：
python analyze_codec_changes.py --work runs/pilot_seen_dose --output runs/pilot_seen_dose/codec_changes_repeat --threads 4 --batch-size 64
python plot_codec_changes.py --analysis runs/pilot_seen_dose/codec_changes_repeat
仅重新生成已有数据的图表：
python plot_codec_changes.py --analysis runs/pilot_seen_dose/codec_changes

目标
冻结一个 LEXA 词典，DeepSeek 只读取文字证据卡、对照状态及药物描述。
无需新增 LLM token、训练 LLM 或监督训练 code editor。
工具执行精确数值解码，DeepSeek 根据工具反馈最多修订两次。
最后独立读取真实处理组评价，agent 从来得不到真实处理组的基因表达。

开始使用
cd /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness

先做全部离线准备（公开数据、划分、pilot LEXA 训练、导出）：
python pipeline.py --stage offline

配置 API 后运行全部阶段，完成的离线产物会自动复用：
export DEEPSEEK_API_KEY='在自己的终端设置，不要把密钥发进聊天'
bash run.sh

默认模型 deepseek-flash 来自本次核对的 DeepSeek 官方文档；可修改：
export DEEPSEEK_MODEL='你要使用的模型名'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
也支持：bash run.sh --api-key-file /path/to/private_key_file --model your_model

第一次真实训练默认 CPU 4 threads、最多30 epochs、validation patience7，
有独立分配的 GPU 时可显式 --device cuda:0。不会自动抢占全部GPU。
模型和词典训练不需要 API。没有 key 的 --stage all 会在 API 阶段前停止，
pipeline_status.json 显示 waiting_for_deepseek_api_key。
依赖见 requirements.txt。当前环境不需安装 OpenAI SDK、anndata 或 wandb。

默认实验规模
主协议：每种药物将一个中间非零剂量整体留出；其余剂量按作者replicate标签
1/2作训练、3作验证；同一培养孔不会跨split。
初轮每种药物选一个 ready 的测试培养孔，共最多4个case；每case处理/匹配对照
均至少20细胞。它是screening，不能当全数据结果。
每轴一次API标注，共32次；默认one_shot和agent两种方法，每case每方法最多
8次API请求，agent最多3次解码评价，one_shot仅评价一次。无隐藏自动API重试。
原始数值基因证据在所有文字卡后仍保留，文字标签无法覆盖真实数值。

完整ready cases + 消融：
bash run.sh --case-limit 0 --methods one_shot,agent,agent_no_labels,shuffled_labels

同一work目录的同一case已完成结果会复用。改变模型/请求/标签/预算会拒绝混用；
做新的重复实验请使用 --work 新目录，保留同样的数据协议。
将4case结果扩为全case可复用已有case并增加新case。

整药物留出的后续实验：
bash run.sh --work runs/leave_Dex --protocol leave_drug_out --holdout-drug Dex
另外三个药物可分别替换。必须重新训练词典并重新标注，不能直接复用见过
被留出处理表达的词典来声称zero-shot。药物机制文本属于外部知识先验。

数据与解释边界
sciPlex2为GEO GSM4150377，A549，四药物，多个µM剂量，固定24h。
不包含时间梯度，不检验时间泛化，也不在本轮检验跨细胞类型/供者迁移。
Dex必须匹配乙醇对照，其他药物匹配各自DMSO零剂量对照。
完整well保持同split；作者triplicate标签不等于已核实的独立生物重复。
严格hash分配过滤后，还执行原始counts表达QC；所有排除留在metadata.csv。
保留唯一versioned Ensembl IDs；symbol仅展示，不把不同同名基因混合。
归一化先在全基因上做CP10K/log1p，然后只按训练集选2000基因。
LEXA保留32轴×16状态×64维球面结构；encoder512/decoder256与
train-only SVD/KMeans scaffold是为本pilot做的适配，不是论文原配置复现。
模型解码效应可以被精确计算；其药物解释、剂量强度和真实生物效应仍需评价。

主要文件
download_data.py：原始GEO下载、CRC/尺寸/hash审计，可续传。
prepare_data.py：QC、按well划分、train-only特征选择、匹配对照及独立evaluation。
fit_lexa.py：复用原LEXA核心训练，冻结后导出精确contributions和control codes。
annotate_cards.py：DeepSeek逐轴标注512状态，保留正负基因证据、检查JSON和基因引用。
agent.py：有请求预算的文字工具调用；最终以实际被工具评价的proposal为准。
tools.py：状态卡、source→target/fraction编辑、冻结解码、数值和分布支持诊断。
experiment.py：生成盲化请求、agent消融、独立真实表达评分。
pipeline.py：编排全部阶段、状态记录和复用产物。
实验设计.txt：研究问题、对照、指标、泄漏约束和文献依据。

核心输出（默认runs/pilot_seen_dose）
prepared/manifest.json：真实样本数、排除数、划分、HVG和输入来源。
prepared/metadata.csv：全部细胞与QC/分组/split。
prepared/evaluation.npz：真实处理组，只有offline score/representation diagnostic读取。
lexa/exact_code_effects.npz：冻结词典的512个基因贡献向量。
lexa/evidence_cards.jsonl：原始有符号基因证据卡，无伪造生物标签。
lexa/labels.json：DeepSeek标注，有effects hash、模型和已完成轴数。
experiment/inputs/：只含对照codes和文字请求，不含真实处理表达。
experiment/<method>/<case>/agent_trace.json：模型文字、工具输入输出和调用记录。
experiment/<method>/<case>/evaluation_*.npz：实际候选的解码变化。
experiment/scores_by_well.csv、scores_by_condition.csv、scores_summary.csv：最终指标。

指标使用“相对匹配对照的表达变化”。no_edit、训练响应均值和同药物最近训练剂量
作为基础对照。top100是观察到的绝对均值变化最大的100个基因，不冒称统计学DEG。
本pilot所有表达指标仅覆盖训练选择的2000基因，不是原始58347特征的全基因组评价。
每条件先平均well指标，再平均条件，不能把单细胞当成独立实验重复计算置信区间。
测试真值不用于prompt选择、状态标注、词典学习或agent候选打分。

离线验证
python -m unittest discover -v
测试使用合成小数据；测试通过不能代替真实药物预测结果。
真实API调用前不会生成假的language labels或假的预测准确率。

2026-10-26 分层占用预测 v3 已完成（开发验证）
详见 V3_README.txt，结果 runs/pilot_seen_dose/experiment_v3_hierarchy_fixed/RESULTS.html。冻结codec，训练响应检索＋轴/state文字标注＋32轴候选＋联合复核，全部16组完成并独立审计。新版RMSE0.212023，旧agent0.349634，同信息表达插值0.194145，code插值0.211953；尚无语言修正增益，joint未改变local结果。原5µM已揭盲，不能作为独立新测试。原pipeline/v2实验继续原样保留；v3通过hierarchy_experiment.py独立命令执行。


原5µM数据的CPA/CODEX横向比较（已完成，2026-10-27）
CPA与CODEX作者模型各3个CPU训练种子，与原16孔/2000genes/786treated/951controls统一评价。
结果：benchmarks/pilot_matched_baselines/RESULTS.html（及RESULTS.txt、scores_summary.csv）。
CODEX直接预测ΔRMSE .195562±.002547，CPA .242347±.013473；简单表达插值.194145，原纯机制LEXA+DeepSeek .356299。
模型对照使用其他剂量真实响应训练；这是已揭盲开发集的同协议比较，不是公开榜单成绩。本轮无新DeepSeek调用。
复现与独立审核：benchmarks/pilot_matched_baselines/README.txt、protocol.json、independent_score_audit.json、completion_manifest.json。
