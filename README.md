# wb_topic_final
2022人民网算法竞赛最终上传版

环境配置见requirements.txt

请先在文件夹下建立dataset文件夹，再在dataset文件夹下新建raw_data文件夹，将train.csv与test.csv（主办方所给数据集）放至该文件夹下

若想直接加载参数生成submission.csv，请先下载网盘里的参数，并解压到目录下的'model_states'文件夹，并运行inference.py即可

若想训练模型，先运行datapreprocess.py以获取训练集与验证集，再运行train.py获得模型参数，请注意修改mfile的路径，或者在项目目录下新建output文件夹，以免报错。

若想验证模型在验证集上效果，请运行merge_predict.py.

参数已放至网盘：

最终的结果由15个模型ensemble生成。
