# wb_topic_final
2022人民网算法竞赛最终上传版

赛题二 飞越阿卡姆队

环境配置见requirements.txt

请先在文件夹下建立dataset文件夹，再在dataset文件夹下新建raw_data文件夹，将train.csv与test.csv（主办方所给数据集）放至该文件夹下,再运行运行datapreprocess.py以获取训练集与验证集

若想直接加载参数生成submission.csv，请先下载网盘里的参数，并解压到目录下的'model_states'文件夹，并运行inference.py即可

若想训练模型，运行train.py获得模型参数，请注意修改mfile的路径，或者在项目目录下新建output文件夹，以免路径报错。

若想验证模型在验证集上效果，请运行merge_predict.py.

参数已放至网盘：

最终的结果由15个模型ensemble生成，15个模型参数由1.修改模型activation=True，修改损失为nn.BCEloss 2.锁定transformer不同层的参数 3.在原来参数基础上加上r-drop/smooth-f1-loss-linear继续训练 4.使用pu_loss 4种方法生成

主要采用的方法是Roberta-wmm + mlp, 损失我们尝试了nn.BCEloss(), 多标签softmax损失函数，具体见：https://spaces.ac.cn/archives/7359/comment-page-1
r-drop（利用dropout的随机性），smooth_f1_loss_linear（针对f1指标的平滑实现）。训练过程中还尝试了冷冻transformer层参数的办法。除了Roberta-wmm外我们还尝试了其他bert，比如ERNIE，最终效果并没有超过Roberta，因此最后没有采用。

数据处理方面，首先去除了网络爬虫带来的一些额外字段，其次去除了表情，并将繁体字转换成中文，为了调用方便，数据存储方式都采取了.json的格式。
