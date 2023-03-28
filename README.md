# NanoDetPlus
<!-- TOC -->

- <span id="content">[NanDet 描述](#NanoDet-描述)</span>
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [运行](#运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#usage)
        - [运行](#running)
        - [结果](#outcome)
    - [模型导出](#模型导出)
        - [用法](#usage)
        - [运行](#running)
    - [推理过程](#推理过程)
        - [用法](#usage)
        - [运行](#running)
        - [结果](#outcome)
    - [模型说明](#模型说明)
        - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo 主页](#modelzoo-主页)

<!-- /TOC -->

## [NanoDetPlus 描述](#content)

NanoDetPlus作为Nanodet的迭代版本，与上一代NanoDet相比，在仅增加1毫秒多的延时的情况下，精度提升了30%。同时NanoDet-Plus改进了代码和架构，提出了一种非常简单的训练辅助模块，使模型变得更易训练！它在COCO2017数据集上mAP（0.5：0.95）检测精度达到了27%，十分轻量化。

[源项目GitHub地址]
https://github.com/RangiLyu/nanodet

## [模型架构](#content)

NanoDetPlus：

[NanoDetPlus 知乎中文介绍](https://zhuanlan.zhihu.com/p/306530300)

## [数据集](#content)

数据集可参考文献：

MSCOCO2017

- 数据集大小：19.3G, 123287张80类彩色图像

    - 训练：19.3G, 118287张图片

    - 测试：1814.3M, 5000张图片

- 数据格式：RGB图像.

    - 注意：数据将在src/dataset.py 中被处理


## [环境要求](#content)

- 硬件（Ascend）
    - 使用Ascend处理器准备硬件环境。
- 架构
    - [MindSpore](https://www.mindspore.cn/install)
- 想要获取更多信息，请检查以下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [脚本说明](#content)

### [脚本和示例代码](#content)

```NanoDetPlus
.
└─NanoDetPlus
  ├─README.md
  ├─ascend310_infer                           # 实现310推理源代码
  ├─scripts
    ├─run_single_train.sh                     # 使用Ascend环境单卡训练
    ├─run_distribute_train.sh                 # 使用Ascend环境八卡并行训练
    ├─run_distribute_train_gpu.sh             # 使用GPU环境八卡并行训练
    ├─run_single_train_gpu.sh                 # 使用GPU环境单卡训练
    ├─run_infer_310.sh                        # Ascend推理shell脚本
    ├─run_eval.sh                             # 使用Ascend环境运行推理脚本
    ├─run_eval_gpu.sh                         # 使用GPU环境运行推理脚本
  ├─config
    └─default_config.yaml                       # 参数配置
  ├─src
    ├─dataset.py                              # 数据预处理
    ├─NanodetPlus.py                          # 网络模型定义
    ├─init_params.py                          # 参数初始化
    ├─lr_generator.py                         # 学习率生成函数
    ├─coco_eval                               # coco数据集评估
    ├─box_utils.py                            # 先验框设置
    ├─_init_.py                               # 初始化
    ├──model_utils
      ├──config.py                            # 参数生成
      ├──device_adapter.py                    # 设备相关信息
      ├──local_adapter.py                     # 设备相关信息
      ├──moxing_adapter.py                    # 装饰器(主要用于ModelArts数据拷贝)
  ├─train.py                                  # 网络训练脚本
  ├─export.py                                 # 导出 AIR,MINDIR模型的脚本
  ├─postprogress.py                           # 310推理后处理脚本
  └─eval.py                                   # 网络推理脚本
  └─create_data.py                            # 构建Mindrecord数据集脚本
  └─data_split.py                             # 迁移学习数据集划分脚本
  └─quick_start.py                            # 迁移学习可视化脚本
  └─default_config.yaml                       # 参数配置
  └─shufflev2_x1.yaml                         # ShufflenetV2_1.0X 预训练权重

```

### [脚本参数](#content)

```default_config.yaml
在脚本中使用到的主要参数是：
"img_shape": [320, 320],                                                            # 图像尺寸
"num_retinanet_boxes": 2125,                                                        # 设置的先验框总数
"nms_thershold": 0.6,                                                               # 非极大抑制阈值
"min_score": 0.05,                                                                  # 最低得分
"max_boxes": 100,                                                                   # 检测框最大数量
"lr_init": 1e-6,                                                                    # 初始学习率
"lr_end_rate": 5e-3,                                                                # 最终学习率与最大学习率的比值
"warmup_epochs1": 2,                                                                # 第一阶段warmup的周期数
"warmup_epochs2": 5,                                                                # 第二阶段warmup的周期数
"warmup_epochs3": 23,                                                               # 第三阶段warmup的周期数
"warmup_epochs4": 60,                                                               # 第四阶段warmup的周期数
"warmup_epochs5": 160,                                                              # 第五阶段warmup的周期数
"momentum": 0.9,                                                                    # momentum
"weight_decay": 1.0e-4,                                                             # 权重衰减率
"num_default": [1, 1, 1, 1],                                                        # 单个网格中先验框的个数
"extras_out_channels": [96, 96, 96, 96],
"extras_out_channels_aux": [192, 192, 192, 192],
"feature_size": [40, 20, 10, 5],                                                    # 特征层尺寸
"aspect_ratios": [[1.0], [1.0], [1.0], [1.0]],                                      # 先验框大小变化比值
"steps": [8, 16, 32, 64],                                                           # 先验框设置步长
"anchor_size":[8, 16, 32, 64],                                                      # 先验框尺寸
"mindrecord_dir": "/cache/MindRecord_COCO",                                         # mindrecord文件路径
"coco_root": "/cache/coco",                                                         # coco数据集路径
"train_data_type": "train2017",                                                     # train图像的文件夹名
"val_data_type": "val2017",                                                         # val图像的文件夹名
"instances_set": "annotations_trainval2017/annotations/instances_{}.json",          # 标签文件路径            
"coco_classes": ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',       # coco数据集的种类           
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush'),
"num_classes": 80,                                                                 # 数据集类别数
"voc_root": "",                                                                    # voc数据集路径
"voc_dir": "",
"image_dir": "",                                                                   # 图像路径
"anno_path": "",                                                                   # 标签文件路径
"save_checkpoint": True,                                                           # 保存checkpoint
"save_checkpoint_epochs": 1,                                                       # 保存checkpoint epoch数
"keep_checkpoint_max":1,                                                           # 保存checkpoint的最大数量
"save_checkpoint_path": "./ckpt",                                                  # 保存checkpoint的路径
"finish_epoch":0,                                                                  # 已经运行完成的 epoch 数
"checkpoint_path":"./ckpt/nanodetplus-497_916.ckpt"                                # 用于推理的checkpoint路径
```

### [训练过程](#content)

#### 用法

运行train.py进行训练。train.py的用法如下：

```训练
# 八卡并行训练示例：

bash scripts/run_distribute_train.sh 8 /data/hccl.json /cache/mindrecord_dir/ /config/default_config.yaml

# 单卡训练示例：

bash scripts/run_single_train.sh 0 /cache/mindrecord_dir/ /config/default_config.yaml

```

#### 运行
[COCO2017数据集下载](https://www.kaggle.com/datasets/aishwr/coco2017)
```cocodataset
数据集结构
└─cocodataset
  ├─train2017
  ├─val2017
  ├─test2017
  ├─annotations

```

```default_config.yaml
训练前，先创建MindRecord文件，以COCO数据集为例，yaml文件配置好coco数据集路径和mindrecord存储路径
# your cocodataset dir
coco_root: /home/DataSet/cocodataset/
# mindrecord dataset dir
mindrecord_dr: /home/DataSet/MindRecord_COCO
```

```MindRecord
# 生成训练数据集
python create_data.py --create_dataset coco --prefix nanodetplus.mindrecord --is_training True --config_path
(例如：python create_data.py  --create_dataset coco --prefix nanodetplus.mindrecord --is_training True --config_path /home/nanodet/config/default_config.yaml)

# 生成测试数据集
python create_data.py --create_dataset coco --prefix nanodetplus_eval.mindrecord --is_training False --config_path
(例如：python create_data.py  --create_dataset coco --prefix nanodetplus.mindrecord --is_training False --config_path /home/nanodet/config/default_config.yaml)
```

```python命令
Ascend:
# 八卡并行训练示例(在nanodetplus目录下运行)：
bash scripts/run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [MINDRECORD_DIR] [CONFIG_PATH] [PRE_TRAINED(optional)] [PRE_TRAINED_EPOCH_SIZE(optional)]
# example: bash scripts/run_distribute_train.sh 8 ~/hccl_8p.json /home/DataSet/MindRecord_COCO/ /home/nanodet/config/default_config.yaml

# 单卡训练示例(在nanodetplus目录下运行)：
bash scripts/run_single_train.sh [DEVICE_ID] [MINDRECORD_DIR] [CONFIG_PATH]
# example: bash scripts/run_single_train.sh 0 /home/DataSet/MindRecord_COCO/ /home/nanodet/config/default_config.yaml
```

#### 结果

训练结果将存储在示例路径中。checkpoint将存储在 `./ckpt` 路径下，训练日志将被记录到 `./log.txt` 中，训练日志部分示例如下：

```训练日志
epoch: 381 step: 458, loss is 3.8455588817596436
lr:[0.047309]
Train epoch time: 155577.828 ms, per step time: 339.690
epoch: 382 step: 458, loss is 3.9784750938415527
lr:[0.046789]
Train epoch time: 155582.377 ms, per step time: 339.700 ms
epoch: 383 step: 458, loss is 4.855691432952881
lr:[0.046268]
Train epoch time: 155616.933 ms, per step time: 339.775 ms
```

- 如果要在modelarts上进行模型的训练，可以参考modelarts的[官方指导文档](https://support.huaweicloud.com/modelarts/) 开始进行模型的训练和推理，具体操作如下：

```ModelArts
#  在ModelArts上使用分布式训练示例:
#  数据集存放方式

#  ├── MindRecord_COCO                                              # dir
#    ├── annotations                                                # annotations dir
#       ├── instances_val2017.json                                  # annotations file
#    ├── checkpoint                                                 # checkpoint dir
#    ├── pred_train                                                 # predtrained dir
#    ├── MindRecord_COCO.zip                                        # train mindrecord file and eval mindrecord file

# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "distribute=True"
#          设置 "keep_checkpoint_max=5"
#          设置 "save_checkpoint_path=/cache/train/checkpoint"
#          设置 "mindrecord_dir=/cache/data/MindRecord_COCO"
#          设置 "epoch_size=550"
#          设置 "modelarts_dataset_unzip_name=MindRecord_COCO"
#          设置 "pre_trained=/cache/data/train/train_predtrained/pred file name" 如果没有预训练权重 pre_trained=""

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/nanodet"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../MindRecord_COCO"(选择MindRecord_COCO文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选择a或者b其中一种方式。
#        a.设置 "enable_modelarts=True"
#          设置 "mindrecord_dir=/cache/data/MindRecord_COCO"
#          设置 "checkpoint_path=/cache/data/checkpoint/checkpoint file name"
#          设置 "instance_set=/cache/data/MindRecord_COCO/annotations/instances_{}.json"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (3) 设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (4) 在modelarts的界面上设置代码的路径 "/path/nanodet"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "../MindRecord_COCO"(选择MindRecord_COCO文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
```

### [推理过程](#content)

#### <span id="usage">用法</span>

使用shell脚本进行评估。shell脚本的用法如下：

```bash
Ascend:
bash scripts/run_eval.sh [DEVICE_ID] [DATASET] [MINDRECORD_DIR] [CHECKPOINT_PATH] [ANN_FILE PATH] [CONFIG_PATH]
# example: bash scripts/run_eval.sh 0 coco /home/DataSet/MindRecord_COCO/ /home/model/nanodetplus/ckpt/nanodetplus_500-458.ckpt /home/DataSet/cocodataset/annotations/instances_{}.json /home/nanodetplus/config/default_config.yaml
```

> checkpoint 可以在训练过程中产生。

#### <span id="outcome">结果</span>

计算结果将存储在示例路径中，您可以在 `eval.log` 查看。

```mAP
Ascend:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.414
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.073
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.252
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.269
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.430
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.190
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.707

========================================

mAP: 0.2641480074063847
```


### [模型导出](#content)

#### <span id="usage">用法</span>

导出模型前要修改config.py文件中的checkpoint_path配置项，值为checkpoint的路径。

```shell
python export.py --file_name [RUN_PLATFORM] --file_format[EXPORT_FORMAT] --checkpoint_path [CHECKPOINT PATH]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]

#### <span id="running">运行</span>

```运行
python export.py  --file_name nanodetplus --file_format MINDIR --checkpoint_path /cache/checkpoint/nanodetplus_510_458.ckpt
```

- 在modelarts上导出MindIR

    ```Modelarts
    在ModelArts上导出MindIR示例
    # (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
    #       a. 设置 "enable_modelarts=True"
    #          设置 "file_name=retinanet"
    #          设置 "file_format=MINDIR"
    #          设置 "checkpoint_path=/cache/data/checkpoint/checkpoint file name"

    #       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
    #          在modelarts的界面上设置方法a所需要的参数
    #          注意：路径参数不需要加引号
    # (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
    # (3) 在modelarts的界面上设置代码的路径 "/path/retinanet"。
    # (4) 在modelarts的界面上设置模型的启动文件 "export.py" 。
    # (5) 在modelarts的界面上设置模型的数据路径 ".../MindRecord_COCO"(选择MindRecord_COCO文件夹路径) ,
    # MindIR的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
    ```

### [推理过程](#content)

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

#### <span id="usage">用法</span>

在推理之前需要在昇腾910环境上完成模型的导出。推理时要将iscrowd为true的图片排除掉。在cpp_infer目录下保存了去排除后的图片id。
还需要修改config.py文件中的coco_root、val_data_type、instances_set配置项，值分别取coco数据集的目录，推理所用数据集的目录名称，推理完成后计算精度用的annotation文件，instances_set是用val_data_type拼接起来的，要保证文件正确并且存在。

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_TYPE] [DEVICE_ID]
```

#### <span id="running">运行</span>

```运行
 bash run_infer_310.sh ./nanodetplus.mindir ./dataset/coco2017/val2017 ./image_id.txt 0
```

#### <span id="outcome">结果</span>

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

```mAP
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.265
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.279
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.074
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.255
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.416
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.430
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.190
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.711
 
========================================

mAP: 0.26522640420473925
```

## [模型说明](#content)

### [性能](#content)

#### 训练性能

| 参数                        | Ascend                                               
| -------------------------- |------------------------------------------------------|
| 模型名称                    | Nanodetplus                                          
| 运行环境                    | Ascend 910；CPU 2.6GHz，192cores；Memory 755G；系统 Euler2.8 |
| 上传时间                    | 26/03/2023                                           |
| MindSpore 版本             | 1.8.1                                                |
| 数据集                      | 123287 张图片                                           |
| Batch_size                 | 32                                                   |
| 训练参数                    | src/default_config.py                                        |
| 优化器                      | Momentum                                             |
| 损失函数                    | Generalized Focal Loss + GIou Loss                   |
| 最终损失                    | 5.1842780113220215                                   |
| 精确度 (8p)                 | mAP[0.264]                                           | 
| 训练总时间 (8p)             | 22h40m15s                                            | 

#### 推理性能

| 参数                 | Ascend                                                 |
| ------------------- |--------------------------------------------------------|
| 模型名称             | NanodetPlus                                            |
| 运行环境             | Ascend 910；CPU 2.6GHz，192cores；Memory 755G；系统 Euler2.8 |
| 上传时间             | 26/03/2023                                             |
| MindSpore 版本      | 1.8.1                                                  |
| 数据集              | 5k 张图片                                                 |
| Batch_size          | 32                                                     |
| 精确度              | mAP[0.264]                                             |
| 总时间              | 18 mins and 40 seconds                                 |
