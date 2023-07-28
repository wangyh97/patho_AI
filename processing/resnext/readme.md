# 端到端模型 -- resnext
## 文件架构
* train.py：从头训练resnext
* load_data.py: dataloader,内设不同数据集平衡方法
* resnext.py：模型部分
* evaluation.ipynb：用于eval的notebook，即时输出结果
## todo
* 新建load_features.py文件，用于其他模型的端对端模型应用
* 建立train_resnext.py文件，优化结果保存模式
* 将resnext.py改为model.py，用于不同模型的finetune