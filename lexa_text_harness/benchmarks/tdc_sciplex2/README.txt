TDC sciPlex2 横向对齐工作区

用户目标：用同一数据、划分、输入规则和评分器，获得可以横向比较的分数，并争取榜单结果。
当前阶段：官方协议审计与数据访问；尚无本地可报榜分数，没有启动新模型训练或DeepSeek调用，没有公开提交。

已确认的公开目标
官方任务：unseen chemical perturbation。
官方榜：ChemCPA，BMS .906±.006，Dex .540±.013，Nutlin .835±.009，SAHA .690±.021，四药平均 .743。
https://tdcommons.ai/benchmark/counterfactual_group/scperturb_drug_SrivatsanTrapnell2020_sciplex2/
一般提交规则要求至少5次独立运行，报告均值及标准差，并使用指定测试集。
https://tdcommons.ai/benchmark/overview/

已完成
- 下载PyTDC1.1.15源码，核对官方PyPI SHA256；不改全局Python环境。
- 检查发布包的loader、split、scorer；用合成数据复现问题，未读取真实处理表达进行评分。
- 查到原论文实验附件的位置，以及正确数据loader的Dataverse文件ID。
- 审核旧LEXA训练/特征/文字标注的可复用边界：现checkpoint见过四药，不能直接报unseen-drug成绩。

当前公开接口不能直接作为榜单原始协议
- 默认dataset是Aissa；官网Dataset Description链接也是Aissa。包中sciPlex2对应9845394。
- 默认cold split在只有四种药时会报错；默认seen选项在合成例中产生空验证/测试集。
- split返回obs元数据；scorer没有正确丢弃数值元数据，合成例的表达R²=-9，却可被元数据推到.9999925。
- 这是当前发布接口的实测问题，不证明论文成绩用到了这些错误；原论文可能使用单独实验脚本。

原始实验附件
2024论文Appendix B.1说明各模型有run_chemical_sc.py：
https://wenhao-gao.github.io/assets/publications/2024_TDC2/paper.pdf
https://drive.google.com/drive/folders/1RlBnRPmWFRQ6M_1EQ_FMwFb1Y8IjXoyC?usp=sharing
当前本机Google连接不可达，web工具也未取到附件；未判定附件是否失效或是否需要权限。
公开发布代码的修复不能替代恢复论文的确切数据/基因/归一化/药物剂量划分/种子/聚合。

文件
vendor/package_provenance.json：版本、来源、SHA256。
protocol_audit/protocol_contract.json：已知/未知比较约定与来源。
protocol_audit/audit_official_api.py：合成数据重现脚本。
protocol_audit/audit_results.json：实际错误及评分污染探针。
protocol_audit/archive_access.json：附件访问状态。
data/：官方scPerturb Zenodo来源的断点数据与元数据；必须完整hash验证后才算下载完成。
download_scperturb_source.py：有时间预算、Range验证与断点恢复的下载器。
maintainer_questions.txt：未发送的复现问题清单。

复现接口审计
/cpfs_speech/user/hejia.hj/code/anaconda/bin/python benchmarks/tdc_sciplex2/protocol_audit/audit_official_api.py

比较准入条件
1. 取得不可变的数据和split清单，锁定gene顺序、表达层、control池、dose汇总和scorer。
2. 在这个协议下先复跑一个已发表baseline，确认数值解释和样本范围。
3. 所有方法使用相同允许训练信息；新fold训练LEXA及标注。不可借用已含测试药物的旧词典。
4. 至少5个独立run；所有预测先封存再评分。主榜指标与附加Δ指标分别报告。
5. 只有完成上述核验，才称官方榜可比。自行制定的统一协议应标为独立复现实验。

2026-10-27 本轮最终进展
- scPerturb Zenodo源文件完整下载并通过发布MD5，145178504bytes；24,262cells×58,347genes，全部A549，CSR float32，无layers/raw。尚未确认与TDC Dataverse副本逐字节相同。
- data/source_metadata_report.json仅检查结构及obs/var，没有载入X表达值或评分；显式control529与各药dose0的3901细胞须按原协议处理，不能自动合并。
- 已检查TDC/PyTDC完整main、2024历史版本、五条相关开发分支及作者公开仓库，仍未恢复run_chemical_sc.py或固定fold。历史loader会包含表达矩阵，当前obs-only问题是后来的回归；不能据此指控原论文实验。
- CPA/CODEX作者完整代码已取得并锁定commit/SHA，原始处理后GSM_new.h5ad仍缺失。保存notebook单次结果不等于论文多随机种子均值。详见../../literature/tdc_leaderboard/CPA_CODEX_reproduction_findings.txt。
- 没有新的训练、DeepSeek调用、模型score或外部提交。正式榜单推进仍需原始实验附件，不能把自行修订协议命名为官方等价。
