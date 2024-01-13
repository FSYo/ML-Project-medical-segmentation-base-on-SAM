# ML-Project-medical-segmentation-base-on-SAM

`train.py` ：训练 decoder 的代码，可选学习参数，decoder任务类型等参数进行训练

`class_decoder.py` : 包装后的 decoder，通过添加一个MLP结构增加了分类功能

`Sam_btcv.py` ：继承sam类包装新的decoder

`utils` ：模块化数据集转换、模型训练、模型测试等工具
