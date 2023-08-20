# <div align="center">IRP - Development of a Tracking System for the Surveillance of Birds in Airports</div>

With the recent development of air travel as a common mean of transport, the question of safety and security in airports has become crucial. Amongst all the different existing danger, birds are considered as one of the main threats to aircraft safety. Bird strikes occur mainly during landing and take off. However, the current surveillance systems are not robust and accurate enough to alert from the presence of birds in restricted areas.

The main purpose of this IRP project is the development of a tracking solution for the surveillance of birds in airports to tackle the problem of birds monitoring. This tracking system aim to track and monitor birds in particular in restricted airports area in order to reduce the risks of bird strikes. The system developed during this project has beed designed to meet the following objectives:

- Track and monitor birds from a static camera position
- Follow the trajectory of any other moving objects
- Robustly handle challenging situations.

The tracking system proposed during this IRP aim to address the problem of birds monitoring in airports. The objective is to include the tracking algorithm as part of a surveillance system in order to alert from the presence of any moving objects, in particular birds. It would therefore increase the safety in airports.

This document describes the different components of this project and their usage. The provided technical work contains the following files and folders:

- ByteTrack: contains the tracking algorithm developed during this project
- Guide_Notebooks: contains two notebooks. One is a demonstration of the tracker's usage and the second one shows how to train a YOLO model with the custom Dataset
- IRP_Dataset: contains the custom dataset created during this project
- siammot: contains the source code for the SiamMOT model. This model is not correctly working
- Tracking_Outputs: contains the videos resulting from tracking on the testing set.

## <div align="center">Installation Instructions</div>

The tracking algorithm has been developed using multiple Python libraries. In order for the tracker to work correctly, make sure the needed packages are installed on your working environment.

Pip install the required packages in a [**Python>=3.8**](https://www.python.org/) environment.

```bash
pip install -r requirements.txt
```

This project is mainly based on [**Ultralytics**](https://github.com/ultralytics/ultralytics) and **OpenCV** packages.

## <div align="center">Dataset</div>

The custom dataset created for this project is provided in the **IRP_Dataset** ZIP file. In order to be used, it should be first unzipped. The dataset folder is organised as follows:

```bash
IRP_Dataset/
├── train/
│   ├── images
│   └── labels
├── YOLO_train/
│   ├── images
│   └── labels
├── val/
│   ├── images
│   └── labels
└── test/
│   ├── Test_01
│   ├── Test_02
│   ├── ...
│   └── Test_15
```

It contains 4 different subfolders used for different tasks. The subfolders **train**, **YOLO_train**, and **val** contain two subsubfolders **images** and **labels**. These two subsubfolders contain folders for each different frame sequence with the frames and the labels files respectively. 

The **YOLO_train** subfolder contains the data used for the training of the [**Ultralytics**](https://github.com/ultralytics/ultralytics) Yolov8x model. It contains two subfolders for the images and the label files respectively. The different frames extracted from different videos are not separated into different folder since the YOLOv8 model is a detection model that does not need to process each frame sequence individually. The label files are in the YOLO format. Therefore, these specific annotations cannot be used directly to train or validate a tracking model.

The **train** subfolder has not been used directly in the project. It contains the same data save in the **YOLO_train** but the frames are stored in different folder in order to distinguish each frame sequence. In addition, the labels files are also in YOLO format, with an additional value corresponding to the tracking ID. This subfolder could be used to train directly a tracking model such as [**SiamMOT**](https://arxiv.org/abs/2105.11595). It is important to note that the frame sequences in this subfolder do not correspond to videos recorded from a static camera position.

The **val** subfolder contains the videos that have been used for the validation of the proposed tracking model. It contains 5 different frame sequences annotated for tracking in the YOLO format. 

The **test** subfolder contains different frame sequences used to visually assess the tracking performance.

In addition, each folder containing a single frame sequence also contains a ```meta_info.txt``` file providing informations about the video.

```bash
[METAINFO]
url: https://www.youtube.com/watch?v=ZO5lV0gh5i4
begin: 00:02:22
end: 00:02:42
FPS: 25
major_class: bird
resolution: (1280.0, 720.0)
```

## <div align="center">Tracking Algorithm</div>

The source code for the tracking system developed during this project is provided in the **ByteTrack** folder. This tracker is based on [**Ultralytics**](https://github.com/ultralytics/ultralytics) Yolov8 model, **OpenCV** background subtraction algorithm, and [**BYTE**](https://arxiv.org/abs/2110.06864) association algorithm. This tracking algorithm can be used as follows:

```python
import os
from glob import glob 
import cv2

from ByteTrack import Tracker # Import the Tracking model

# Load a model
model = Tracker("ByteTrack/weights/trained_yolov8x.pt")

# Use the model
metrics = model.val("IRP_Dataset/val")  # evaluate model performance on the validation set

results = model("IRP_Dataset/test/Test_01", fps=25)  # predict and track on an entire video or frame sequence

# Predict and track frame by frame
# Get the path to a video or image sequence
data_path = "/IRP_dataset/test/Test_01"

# Get the fps saved in the same folder
with open(os.path.join(data_path, "meta_info.txt"), "r") as f:
    meta_info = [line[:-1] if line[-1] == '\n' else line for line in f.readlines()]
    
fps = int(meta_info[4].split(" ")[-1])

# List of frames
lst_frames = sorted(glob(os.path.join(data_path, '*.jpg')))

for file in lst_frames:
    frame = cv2.imread(file)

    results = model.track(frame, persist=True, fps=fps)
```

Refer to the Demonstration Notebook for an example of the tracking pipeline.

In addition, the **siammot** folder contains the source code for a second tracking model [**SiamMOT**](https://arxiv.org/abs/2105.11595). This model did not end up working but the code could be useful in some way for further works.

## <div align="center">Tracking Outputs</div>

The videos resulting from tracking on the testing set are available in the folder **Tracking_Outputs**. These videos visually show the tracking performance of the current tracking algorithm and could be used for comparison.

![frame1](https://github.com/theolange01/IRP/assets/116893751/e573920c-863b-4bfa-a78a-bf2432a9c940)
