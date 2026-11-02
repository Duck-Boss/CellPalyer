用户要求：在原来已经做过的5µM开发实验上重跑其他扰动模型，获得横向可比分数。
状态：CPA/CODEX各3seed真实训练及预测完成，所有预测先全局seal再评分。独立审核见对应audit.json。
主要结果：RESULTS.html、RESULTS.txt、scores_summary.csv。详细sample定义见protocol.json。
复现环境：/cpfs_speech/user/hejia.hj/code/anaconda/bin/python；没有安装或改动全局依赖。
项目根目录下命令：
python benchmarks/pilot_matched_baselines/prepare_matched.py
python benchmarks/pilot_matched_baselines/run_cpa.py --seed 0 --output <新目录>
python benchmarks/pilot_matched_baselines/run_codex.py --seed 0 --output <新目录>
两个训练脚本对seed1/2重复；线程环境OMP/OPENBLAS/MKL_NUM_THREADS=4，CPU显式指定。
已有输出应直接审核，无需重复训练；训练脚本拒绝覆盖完成预测。
python benchmarks/pilot_matched_baselines/score_baselines.py verify
python benchmarks/pilot_matched_baselines/report_matched_baselines.py
首次score_baselines.py report因pandas Series.view名称冲突失败；保留该已封存源码，独立报告脚本修正展示，不改预测或评分。
