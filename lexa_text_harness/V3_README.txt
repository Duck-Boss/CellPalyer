LEXA 分层占用预测 v3：运行与解释说明
文档日期：2026-10-26，Asia/Shanghai

状态（编写时核实）
训练检索库、32轴/512状态证据卡、16个case的候选占用已准备，首次输入独立审计通过。
修复后的 experiment_v3_hierarchy_fixed 已完成全部16组local/joint预测，已封存、评分并通过独立审计。
首次 experiment_v3_hierarchy 在轴标注阶段失败：共8次API请求，24轴完成，末批连续3次把gene symbol填到需要versioned Ensembl ID的字段，因此停止；没有真实扰动预测或评分。原始请求、响应、失败状态和源码快照保留在 runs/pilot_seen_dose/experiment_v3_hierarchy/。
本轮修复明确提示允许的gene IDs，并提供具体校验反馈；同时加强中断尝试恢复及观察code数组SHA检查。使用新输出目录完整重跑，不混合旧协议结果。
真实结果已完成：hierarchical_local/joint ΔRMSE均0.212023、ΔPearson均0.740271；code插值0.211953/0.740438；同库表达插值0.194145/0.760182。新版相对旧agent误差降低39.36%，但未超过检索基线。

工作目录
/cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness

工作数据
/cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose

本次输出
/cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose/experiment_v3_hierarchy_fixed

1. 本版实际实现

沿用已冻结的 pilot LEXA best.pt 和 exact_code_effects.npz，32轴×16状态、2000训练选择基因。不重训codec，不微调DeepSeek，不新增LLM token；DeepSeek只接收文字/JSON，完整矩阵运算由本地工具完成。

流程：训练响应检索 → 轴功能文字与轴内状态证据 → 4批逐轴选择（每批8轴，覆盖全部32轴）→ 一次全局反馈修正 → 冻结输出 → 单独评分。

轴卡先从训练mask及随状态变化的基因构建；axis功能说明由DeepSeek分4批生成。状态文字复用此前冻结的512条DeepSeek标签，并展示轴内均匀参考下的正负贡献。轴功能是暂定语言解释，未做新的正式GO富集或人工生物学审核；命名不等于实测通路活性，state编号不代表强度顺序。

2. 训练响应库与评测范围

检索库仅使用5285个训练细胞中的合法处理/对照。处理孔按同drug、vehicle、plate、作者replicate匹配其零剂量对照；处理与对照均须至少20细胞。共91个非空训练处理孔：61个合格、30个低样本排除，得到21个药物剂量条件。每个条件的响应先按孔求差，再对合格孔等权平均，不按细胞数加权。

目标为4药物 BMS/Dex/Nutlin/SAHA 的5µM、24小时、A549：全部16个ready孔，每药4孔，合计786个处理细胞。5µM真实结果此前已经揭盲，所以本轮是开发验证；不能作为新的独立泛化测试。预测器读取合法匹配对照，不读取5µM处理表达；只有全部预测封存后score阶段读取处理真值。

1/10µM训练参考的有效孔数：BMS 4/1、Dex 1/2、Nutlin 4/3、SAHA 3/3。部分参考仅一孔，作者replicate也未被证实代表独立生物重复。固定24h不支持时间泛化结论。

3. 候选空间与baseline

每轴从7个预先构造的目标占用分布中选一个：
- keep：保持当前对照p0。
- nearest：加上同药在自然对数剂量距离最近的训练条件Δp。
- interpolate：加上同药上下两个训练条件在自然对数剂量上的线性插值Δp。
- lower：使用低剂量端点Δp。
- upper：使用高剂量端点Δp。
- half：使用0.5倍插值Δp。
- onehalf：使用1.5倍插值Δp。

所有非keep候选均将p0+Δp逐轴欧氏投影到“非负且16状态和为1”的概率单纯形，保存投影L1改变量。目标超出训练剂量范围时钳制到端点，不进行外推；最近剂量在相同对数距离时选较低剂量。

本版取消最多3轴限制；32轴均可选候选。但它仍是有限候选选择：不能任意生成512个概率，也没有实现任意source→target编辑。若某轴多个候选完全相同，记录名称不同不代表表达预测真的改变。

比较方法：
- expression_nearest / expression_interpolate：使用同一61孔库的真实训练表达Δ，直接检索或插值。
- code_nearest / code_interpolate：所有轴统一采用对应占用候选，然后精确解码。
- hierarchical_local：4批局部提案直接合并。
- hierarchical_joint：在同一局部提案上增加一次全局反馈调用，允许修订32轴选择。
- no_edit和旧v2方法：仅在最终评分中作为历史对照展示。

新表达与code检索对照的训练信息和孔资格一致。旧nearest使用非空训练孔及log1p剂量距离，单独标记，不能混称为完全相同基线。

local和joint共享4次局部成功调用，joint额外一次成功调用；每阶段最多2次格式/传输修复重试。二者不是预算匹配实验。相对旧v2，检索信息、编辑空间和文字结构同时变化，不能把总提升全部归因于层级标注或语言。

4. CLI（逐行执行；凭据不出现在命令内容中）

准备输入；相同来源和源码时可安全复用：
/cpfs_speech/user/hejia.hj/code/anaconda/bin/python /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/hierarchy_experiment.py prepare --work /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose --output /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose/experiment_v3_hierarchy_fixed

真实预测或同协议续跑；默认deepseek-flash，2个case并发：
/cpfs_speech/user/hejia.hj/code/anaconda/bin/python /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/hierarchy_experiment.py predict --work /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose --output /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose/experiment_v3_hierarchy_fixed --api-key-file /cpfs_speech/user/zhoutianhong.zth/.secrets/deepseek.key --workers 2

全部预测封存后评分；无API调用：
/cpfs_speech/user/hejia.hj/code/anaconda/bin/python /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/hierarchy_experiment.py score --work /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose --output /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose/experiment_v3_hierarchy_fixed

重建结果图和HTML；无API调用：
/cpfs_speech/user/hejia.hj/code/anaconda/bin/python /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/hierarchy_experiment.py report --work /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose --output /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose/experiment_v3_hierarchy_fixed

凭据仅从指定文件或DEEPSEEK_API_KEY读取，不要粘贴到日志、README或聊天。更换model/base-url/workers、提示、输入、源码或预算会改变协议。要试新协议，应选择新的--output目录，保留已有结果与失败记录；不要删除旧轨迹伪装成首次尝试。

5. 断点与冻结

prepare会核对来源/实现指纹和已保存产物SHA。predict会验证protocol、标注和各预测阶段的指纹；已完成的axis_annotations.json或annotation batch在指纹相同时直接复用，不再次调用API。已完成的预测阶段同样复用；未完成阶段可能继续消耗尚余预算。若重试预算已经耗尽，保留错误记录并排查，不无限重试。

prediction_seal.json只有全部16case所需输出完成后才生成。score先校验封存文件和源码SHA，再打开处理结果；缺失预测或内容变化会阻止评分。启动预测后，尤其封存后，不要修改primary source、输入、protocol、标注或预测数组。若需要修改实现，使用新输出目录重新建协议与输入，保留旧版。

6. 数值反馈的含义

表达单位：训练全基因CP10K归一化后log1p空间，报告只覆盖选定2000基因。contributions和解码Δ属于模型在该空间的贡献/变化，不是fold change。

精确群体均值变化：delta[g] = sum_a sum_s (q[a,s]-p0[a,s])*u[a,s,g]。它只需要每轴边际占用，不能唯一确定全细胞32轴联合分布，也不识别同一细胞的前后转移。

delta_l2：2000基因变化向量的欧氏范数。
standardized_delta_rms：sqrt(mean((delta/max(train_gene_SD,0.05))²))，是训练SD单位的幅度。
axis TV：0.5*sum_s(abs(q-p0))，范围0到1；阈值计数是描述性变化量，不是显著性检验。
projection_l1：可行占用q与未投影p0+Δp的绝对差之和；它描述约束修正，不是预测误差。
prior_comparisons和interpolated_training_expression_prior：与训练响应先验的余弦/幅度一致性，不是目标处理准确率。
largest_cross_axis_cancellation：不同轴效应相互抵消的诊断；抵消本身不一定是错误。
train_pair_support_diagnostic：在“各轴按边际独立抽样”的计算假设下，平均负对数训练轴对频率（自然对数，nats；0.5伪计数平滑）。它受药物/剂量/细胞状态等混杂，不证明真实联合细胞分布或因果兼容性。

最终主要指标：相对匹配对照的ΔRMSE、ΔPearson、响应L2幅度比；辅以各轴占用TV误差、主导code命中及改变轴数。每孔计算指标，药内孔等权，再四药等权。Top100指最大观测均值变化基因，不是统计学DEG。no_edit的ΔPearson未定义，不应解释为零相关。

7. 主要产物

inputs/manifest.json：16case路径、输入与源码指纹。
inputs/training_library.json/.npz：训练条件、孔资格、完整well cell IDs、等权占用与表达Δ。
inputs/training_wells.npz：逐孔匹配差值。
inputs/axis_cards.json：训练基因证据与冻结状态标签。
inputs/<case_slug>/case.npz、context.json：7候选全精度数组与文字输入。
independent_input_audit.json：训练来源、对照匹配、候选和基因效应独立复算。
axis_annotations.json、axis_cards_annotated.json：本轮暂定轴功能说明。
annotation_requests/、predictions/<case_slug>/requests/：请求、响应和修复记录。
runs/pilot_seen_dose/experiment_v3_hierarchy/source_snapshot/：首次失败版本源码；hierarchy_experiment.py已按原protocol SHA256精确恢复。
predictions/<case_slug>/：code_nearest/code_interpolate/local/joint预测及反馈。
protocol.json、prediction_seal.json：固定实验配置与预测封存。
scores_by_well.csv、scores_by_condition.csv、scores_summary.csv：表达指标。
occupancy_by_well.csv、occupancy_by_condition.csv、axis_changes_by_condition.csv：code分布指标。
evaluation_scope.json、api_usage.csv、RESULTS.html、RESULTS.txt、metrics_comparison.png：评测范围、调用和最终报告。

结果状态：已完成。报告 RESULTS.html / RESULTS.txt，逐孔和逐条件CSV，以及independent_input/prediction/score/execution_audit.json全部在fixed目录。512次局部选择510interpolate+2half；联合阶段16组均无改动。fixed真实API84次，首次失败8次另保留；当前没有运行任务。

报告补充审计结论（先运行report，再执行）：
/cpfs_speech/user/hejia.hj/code/anaconda/bin/python /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/summarize_hierarchy_run.py --output /cpfs_speech/user/zhoutianhong.zth/project/lexa_text_harness/runs/pilot_seen_dose/experiment_v3_hierarchy_fixed
