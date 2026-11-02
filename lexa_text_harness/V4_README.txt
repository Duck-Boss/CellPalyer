纯机制与code文字预测 v4

研究问题：DeepSeek能否仅凭一般药物机制、未经处理的细胞状态和冻结code含义，预测药物诱导的表达变化？

本版使用独立入口pure_mechanism_experiment.py。原pipeline.py、run.sh和active_experiment.json仍对应旧v2，不能互换。
输出目录：runs/pilot_seen_dose/experiment_v4_pure_mechanism_fixed
原experiment_v4_pure_mechanism因机制文献引用ID与校验器不一致中止，保留所有请求/源码；没有评分。

方法
1. 冻结已有LEXA编码器、32×16词典、加法解码器及训练来源的轴/state文字。
2. 用PubMed主源给出四药靶点、作用和适用条件，不给出实验响应、基因变化答案、axis/code推荐或剂量插值。
3. 每例先1次DeepSeek调用提出机制假设，尚不展示code编号。
4. 4批×8轴，比较每轴16个状态的有符号文字/数值证据与当前control，选择shift、keep或unknown。
5. shift选择1至3个目标state形成r；q=(1-alpha)p0+alpha*r。keep/unknown维持p0。所有32轴都可编辑。
6. 本地计算精确基因Δ=sum_as(q-p0)*u，1次联合复核只检查与模型自己提出的机制假设是否一致，允许不改。
7. alpha=.3预设主结果；.1/.6保留为敏感性。后两档复用按.3反馈形成的同一决策，没有按答案调优。

四组相同名义API预算：full / no_mechanism / no_semantics / neither。
隐藏机制时去掉药名、靶点、来源URL及溶剂，只有opaque病例ID；隐藏语义时同时去掉卡片和反馈里的基因身份与功能文字。
匿名基因编号随机排列，不暴露按真实基因排序的映射。无语义组不提供假设基因与匿名数值特征的对应反馈。
同样允许的调用数不等于相同token量；隐藏语义后输入更短。

预测可见：exact_code_effects.npz、baseline.npz、labels.json、药物无关的旧轴标注、病例选择元数据、mechanism_cards_v4.json。
预测不可见：训练配对药物响应/Δp/表达Δ、nearest/interpolate等响应候选、evaluation.npz、codec_changes答案、oracle拟合结果。
manifest保留内部case_id用于编排，模型消息由白名单构建，不能将整个manifest/bundle直接发送。

透明数值对照：no-edit；literal_keyword（明确程序词面+方向匹配，无法匹配则弃权）。后者目前只匹配Nutlin的p53-up轴，是弱规则控制，不代表完整通路推理方法。

数据与评价
同一批已揭盲5µM开发孔：16孔、4药各4孔、786处理细胞、951唯一匹配control、A549、24h、2000训练选择基因。
这不是786个独立扰动实验，也不是新的独立测试。作者replicate是否独立生物重复未核实。
固定codec曾用其他剂量训练；本实验限制的是推理时的信息，不能声称表示从未接触扰动。
实际单细胞不前后配对，预测和评价为群体均值/轴边际；边际q不确定完整单细胞联合分布。
先完成64个case-arm并封存全部输入/输出/源码，再打开真实处理表达评分。
主指标：每孔2000基因RMSE(预测Δ,真实处理均值−对照均值)，药内等孔权，再四药等权。单位log1p(CP10K)。
另报Pearson、cosine、观察绝对变化最大100基因的方向准确率、预测/实际L2幅度、占用TV误差、主导code更换轴数。
Top100不是统计差异表达筛选。零预测的相关性未定义，保留空值/NaN。
answer_codec_diagnostic使用真实答案编码解码，仅为诊断，不是预测或严格下界。
raw_decoded_absolute_by_well.csv使用解码control作为起点，与observed-control锚定的主ΔRMSE不同。

复现（在本项目目录，Python使用 /cpfs_speech/user/hejia.hj/code/anaconda/bin/python）
环境：OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4。
python pure_mechanism_experiment.py prepare --output runs/pilot_seen_dose/experiment_v4_pure_mechanism_fixed
python pure_mechanism_experiment.py predict --output runs/pilot_seen_dose/experiment_v4_pure_mechanism_fixed --api-key-file /cpfs_speech/user/zhoutianhong.zth/.secrets/deepseek.key --workers 4
python review_pure_mechanism.py --work runs/pilot_seen_dose --experiment runs/pilot_seen_dose/experiment_v4_pure_mechanism_fixed --require-predictions
python pure_mechanism_experiment.py score --output runs/pilot_seen_dose/experiment_v4_pure_mechanism_fixed
python review_pure_mechanism_scores.py --output runs/pilot_seen_dose/experiment_v4_pure_mechanism_fixed
python pure_mechanism_experiment.py report --output runs/pilot_seen_dose/experiment_v4_pure_mechanism_fixed
python summarize_pure_mechanism_run.py --output runs/pilot_seen_dose/experiment_v4_pure_mechanism_fixed

完整缓存及source/input/model绑定用于恢复，不应改动已封存预测或按效果选样本。
请求先写盘；异常中断未知请求计入预算；每stage最多3次attempt。耗尽则实验不完整、拒绝评分，不静默丢样本或改成no-edit。
引用格式失败后的fixed run全64组重新开始；原请求不混入新预测。

结果入口
RESULTS.html / RESULTS.txt：指标表与图。
INTERPRETATION.txt / experiment_summary.json：消融、API执行记录与解释限制。
protocol.json / prediction_seal.json：预测前冻结协议及评分前封存。
independent_prediction_audit.json / review_scores.json：独立复核（文件实际名称以审计命令输出为准）。
predictions/<arm>/<opaque_case>/：原始request/response/status、hypotheses、local/joint决策和全部alpha数组。

已验证最终结果（2026-10-27）
完整方案alpha=.3 ΔRMSE .3562988015，不编辑 .3557237162，差+0.0005750853（约差0.162%）。
literal规则 .352798；隐藏机制 .386305；隐藏code语义 .396323；两者隐藏 .372037。
full平均shift13.625/32轴；联合复核使ΔRMSE由.358018降为.356299，但仍未超过no-edit。
响应L2只达实际23.23%，方向也弱，不能归结为仅幅度不足。
敏感性alpha=.1/.6分别.353650/.374897，未替换预设主alpha。
相关性有效范围full15/16、neither10/16、literal4/16仅Nutlin；原汇总跳过未定义值，不能按相同样本范围直接比较。
fixed402请求=384成功阶段+18次格式重试，全部64case-arm完成。原失败run67请求/63响应/4未知，保留。
23项针对测试通过；128预测×3alpha与402真实请求独立审计通过，Δ重算差<=2.22e-15；464评分行独立复算差<=6.7e-16。
