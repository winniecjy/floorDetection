# floorDetection: 设计稿楼层识别分割
从长设计稿上识别多个楼层进行分割，得出分割区域坐标。    
## 目录结构
```python
.
├── workspace # 模型训练环境
│   ├── demo            # 模型效果展示demo
|   |
│   ├── images          # labelimg标记图片   
│   │   ├── test          # 测试集 
│   │   └── train         # 训练集
│   │
│   ├── process         # 数据集处理脚本
│   │   ├── csv2record.py # csv转TFRcords Format
│   │   └── xml2csv.py    # xml合并到csv
│   │
│   ├── output          # 脚本输出文件
│   │   ├── floors.pbtxt  # label标记文件
│   │   ├── test.csv
│   │   ├── test.record
│   │   ├── train.csv
│   │   └── train.record
│   │
│   ├── training       # 训练环境
│   ├── floors_inference_graph       # 训练模型导出目录
│   ├── train.py       # 训练脚本
│   └── export_inference_graph.py    # 训练模型导出脚本
├── models             # tensorflow已实现的各种模型，训练辅助环境（https://github.com/tensorflow/models.git）
└── README.md

```
- `images`
  - `*.jpg`: 设计稿
  - `*.xml`：labelimg产生的标记数据
- `*.record`：tensorflow推荐的标准数据格式，tfrecord数据文件是一种将图像数据和标签统一存储的二进制文件
- `*.pbtxt`：标签映射文件，训练模型过程中也会产生