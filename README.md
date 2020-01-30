# floorDetection: 设计稿楼层识别分割
从长设计稿上识别多个楼层进行分割，得出分割区域坐标。    
## 目录结构
```python
.
├── images           # labelimg标记图片
│   ├── test            # 测试集
│   └── train           # 训练集
├── processing       # 数据集处理脚本
│   ├── csv2record.py   # csv转TFRcords Format
│   └── xml2csv.py      # xml合并到csv
└── models           # tensorflow已实现的各种模型，训练环境（https://github.com/tensorflow/models.git）

```