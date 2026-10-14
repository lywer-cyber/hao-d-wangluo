# 论文素材：当前完成的基线快照

这不是最终论文全部结果。当前主表仅有Restormer-SR与SwinIR，均为seed10；RCAN等未完成结果不填数、不画成零。

## 文件

- `tables/baseline_test_results.csv`：500张测试图的聚合均值，来自原始`metrics.json`。
- `tables/baseline_inventory.csv`：已完成、正在训练及未开始的模型索引。
- `tables/restormer_validation_curve.csv`、`swinir_validation_curve.csv`：5000～60000步、每5000步一次的正常训练期验证记录，日志精度为四位小数。
- `tables/restormer_sensitivity.csv`：27个单参数扫描点，每点50张图。
- `tables/visual_sources.csv`：10张固定展示样本对应的源文件和原生尺寸。
- `figures/*.png/.pdf`：训练曲线、主指标概览和敏感性曲线；PDF为矢量图。
- `figures/fixed_samples_overview.png`：全图缩略预览，不代替原生分辨率BMP。
- `provenance.json`：所有输入素材的SHA256、绘图软件版本与口径说明。

## 复建

在安装`environment/requirements-paper.txt`后，运行仓库中的`build_paper_assets.py`。此步骤不需要完整数据、权重或GPU。

## 使用边界

- 不能将单种子均值写成3种子均值±标准差。
- 不能把50张敏感性扫描结果当作500张主测试结果。
- 没有历史500张逐图浮点指标，不生成逐图显著性检验。
- 不在缩略图上标注由8bit BMP重算后冒充原浮点口径的PSNR。
- 后续最终方法确定后，需统一导出其结果，再制作完整方法对比、固定ROI放大图与误差图。
