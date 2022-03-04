# INSTALL

### 1. Create and activate conda environment:
```
conda create -n labelimg3d python=3.6.2
activate labelimg3d
```

### 2. Clone the repository:
```
git clone https://github.com/stjuliet/Labelimg3D
```

### 3. Install the requirements:
```
cd ~/bbox3d_annotation_tools
pip install -r requirements.txt
```

### 4. Download [yolov4 models](https://github.com/stjuliet/Labelimg3D/releases/tag/yolov4-model-highway) and organize model and dataset files as follows:
```
bbox3d_annotation_tools                             # root directory
    ├── model_us_yolov4                             # model root directory
        ├── obj.names                               # classes
        ├── yolov4.cfg                              # config file
        ├── yolov4_best.weights                     # yolo weights
    ├── dataset_demo                                # dataset root directory
        ├── calib
            ├── {frame_id}_calibParams.xml          # 3×4 calib matrix
        ├── {frame_id}.jpg                          # image
```

### 5. run `main.py`.
```
python main.py
```