 PyTorch 相关测试
相关代码存放于 code/Pytorch 目录下，测试前请先进入该目录：
1. 模型训练测试：在终端执行 `python exp2.py` 命令即可启动模型训练流程。若使用 GPU 进行训练，约90-120min后，训练完成的最终模型文件会自动保存至 ./point 目录中。
2. 趋势图绘制测试：在终端执行 `python play.py` 命令可生成指标变化趋势图，生成的图片文件（test_acc.png）将保存于 Pytorch 目录下。若需调整图表数据，直接修改文件内的 `test_acc` 数据列表即可。


 CNN 图像检索测试
相关代码存放于 code/CNN 目录下，测试前请先进入该目录：
1. 主流程测试：在终端执行`set KMP_DUPLICATE_LIB_OK=TRUE` ，`python main.py`两个 命令即可运行完整检索流程。其中，数据集图像存放于 Data 目录，待检索的查询图像存放于 Query_image 目录；输出内容分为三类：Feature_xxx 目录（存储通过 xxx 模型提取的数据集图像特征）、Query_feature 目录（存储查询图像的特征数据）、Output 目录（存储最终的图像检索结果）。
2. 特征提取单独测试：若需验证各模型的特征提取效果，可在对应 `extract_feature_xxx.py` 文件内自行添加测试代码，通过自定义测试逻辑完成验证。