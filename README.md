# Door Handle Detection using YOLOv12

This project performs object detection for **door handles**, combining datasets of **car door handles** and **general indoor/outdoor handles**. The model is trained using the YOLOv12 object detection framework.

---

## Dataset

The dataset merges:

- **Car door handles** (manually annotated)
- **General door handles** from a Roboflow public dataset


---

## Model Details

- **Model**: YOLOv12n
- **Training Resolution**: 640×640
- **Epochs**: 50
- **Loss**: Default YOLOv12 loss
- **Augmentations**: Mosaic, HSV, random scale
- **Confidence Threshold**: 0.6

---

## Inference

To test on a single image:

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model('door.png', conf=0.6)      # Image to be tested
results[0].show()
results[0].save(filename='output.jpg')
```

The `best.pt` file is in the runs directory, both for 50 and 70 epochs. 