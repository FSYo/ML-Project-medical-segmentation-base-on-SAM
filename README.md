# ML-Project-medical-segmentation-base-on-SAM

`train.py` ：训练 decoder 的代码，可选学习参数，decoder任务类型等参数进行训练

`class_decoder.py` : 包装后的 decoder，通过添加一个MLP结构增加了分类功能

`Sam_btcv.py` ：继承sam类包装新的decoder

`utils` ：数据集转换、模型训练、模型测试等工具

训练时请将json, img, label均放入/dataset文件夹里

## TODO

- [ ] 模型在分类任务上的表现test
- [ ] 评估模型性能
- [ ] 实验报告
