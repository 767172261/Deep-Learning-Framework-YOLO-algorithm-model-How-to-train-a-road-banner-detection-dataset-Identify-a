深度学习框架YOLO算法模型如何训练道路横幅检测数据集 通过训练的YOLOV8的横幅数据集的权重 推理识别道路横幅
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cf66094a140b422788a926fa8f293bb2.png)

@[toc]

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c3f0f32f04c8448baddbc9ffa202f89c.png)


**数据集描述：
道路横幅检测数据集
说明：
2007张，txt格式
训练集验证集按1604:403划分，可直接用于目标检测训练

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b48049e8d4e416c9532b24f3377d123.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d8ffc7888de94126a8cba4b321c62b7e.png)
# 🚩 道路横幅检测数据集 YOLOv8 完整训练与部署指南

---

## ✅ 一、环境搭建

### 1. 安装 CUDA 驱动（适用于 NVIDIA GPU）

确保同学你的系统已经安装了支持 CUDA 的 NVIDIA 显卡驱动。使用以下命令查看驱动版本：

```bash
nvidia-smi
```

如果没有安装，请前往 [NVIDIA 驱动下载页面](https://www.nvidia.com/Download/index.aspx) 下载并安装。

---

### 2. 安装 Anaconda

如果你尚未安装 Anaconda，可以从 [Anaconda 官网](https://www.anaconda.com/products/distribution) 下载并安装。

创建一个用于 YOLOv8 的 Python 虚拟环境：

```bash
conda create -n banner_detection python=3.9
conda activate banner_detection
```

---

### 3. 安装必要依赖项

在激活的虚拟环境中安装以下依赖：

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib numpy labelimg
```

---

## 📁 二、数据集结构说明

道路横幅检测数据集包含 **2007 张图像**，仅有一个类别：

- 类别：`banner`
- 标注格式：YOLOv5/8 支持的 `.txt` 格式
- 已划分为 `train` 和 `val` 集合（1604:403）

### 数据集结构应如下：

```
banner_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### `data.yaml` 内容示例：

```yaml
train: ./banner_dataset/images/train
val: ./banner_dataset/images/val

nc: 1
names: ['banner']
```

---

## 🧠 三、YOLOv8 模型训练流程

### 1. 下载 YOLOv8 官方预训练模型（推荐）

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### 2. 开始训练（Python API）

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8s.pt')

# 开始训练
results = model.train(
    data='banner_dataset/data.yaml',  # 替换为你的 data.yaml 路径
    epochs=100,
    imgsz=640,
    batch=32,              # 根据显存调整，建议从16或32开始
    device=0,              # 使用 GPU 0
    project='runs/banner',  # 结果保存路径
    name='exp',            # 实验名称
    save=True,
    save_period=10,        # 每10个epoch保存一次
    pretrained=True,
    verbose=True
)
```

### 或使用命令行方式训练：

```bash
yolo task=detect mode=train model=yolov8s.pt data=banner_dataset/data.yaml epochs=100 imgsz=640 batch=32 device=0
```

---

## 🔍 四、模型推理（Inference）

### 1. 单张图像推理

```python
from ultralytics import YOLO
import cv2

# 加载训练好的模型权重
model = YOLO('runs/banner/exp/weights/best.pt')

# 推理图像
img_path = 'banner_dataset/images/val/banner_0001.jpg'
results = model(img_path)

# 可视化结果
for r in results:
    im_array = r.plot()
    im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
    cv2.imshow('Detection Result', im)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. 视频流推理（本地视频或摄像头）

```python
cap = cv2.VideoCapture('banner_video.mp4')  # 或使用 0 表示摄像头

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        cv2.imshow('Banner Detection', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 五、模型评估（Validation）

使用验证集评估模型性能：

```python
from ultralytics import YOLO

# 加载最佳权重
model = YOLO('runs/banner/exp/weights/best.pt')

# 执行验证
metrics = model.val(data='banner_dataset/data.yaml')

# 打印关键指标
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
```

---

## 📈 六、结果可视化与分析

YOLOv8 会自动将以下内容保存到 `runs/banner/exp/val/` 目录：

- `confusion_matrix.png`：混淆矩阵
- `precision_recall_curve.png`：PR曲线
- `results.csv`：各 epoch 的损失和指标

你也可以使用 OpenCV 或 Matplotlib 自定义绘制这些图表。

---

## 🧩 七、系统部署建议

### 1. 构建 Web API 服务（可选）

使用 `FastAPI` 或 `Flask` 提供 HTTP 接口，用于上传图像并返回检测结果。

### 示例 FastAPI 代码：

```python
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO('runs/banner/exp/weights/best.pt')

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', im)
        return {"result": buffer.tobytes().hex()}
```

启动服务：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---


---

## 🧾 附录：YOLOv5 与 YOLOv8 模型性能对比

| 模型版本 | mAP@0.5:0.95 |
|----------|--------------|
| YOLOv5（100轮） | **0.909** |
| YOLOv8（100轮） | **0.878** |


