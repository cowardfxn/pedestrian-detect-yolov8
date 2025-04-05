# pedestrian-detect-yolov8

### 官方DeepSORT库
pip install git+https://github.com/nwojke/deep_sort.git
###  # YOLOv8最新代码
pip install git+https://github.com/ultralytics/ultralytics.git


## Remote repo
https://github.com/cowardfxn/pedestrian-detect-yolov8

### YOLOv8 Model performances

Model | Resolution | Process time per frame
:---- |:---- | :--------
yolov8n.pt | 640, 288 | 200ms on average
yolov8l.pt | 640, 288 | 1500ms on average (R&W)
yolov8l.pt | 640, 288 | 350ms on average (Read only)

## Start up

```cmd
python .\pedestrians-detect.py
```

