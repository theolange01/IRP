# <div align="center">IRP - Development of a Tracking System for the Surveillance of Birds in Airports</div>

<p style="text-align: justify;">
With the recent development of air travel as a common mean of transport, the question of safety and security in airports has become crucial. Amongst all the different existing danger, birds are considered as one of the main threats to aircraft safety. Bird strikes occur mainly during landing and take off. However, the current surveillance systems are not robust and accurate enough to alert from the presence of birds in restricted areas.

<p style="text-align: justify;">
The main purpose of this IRP project is the development of a tracking solution for the surveillance of birds in airports to tackle the problem of birds monitoring. This tracking system aim to track and monitor birds in particular in restricted airports area in order to reduce the risks of bird strikes. The system developed during this project has beed designed to meet the following objectives:

- Track and monitor birds from a static camera position
- Follow the trajectory of any other moving objects
- Robustly handle challenging situations.
<p style="text-align: justify;">
The tracking system proposed during this IRP aim to address the problem of birds monitoring in airports. The objective is to include the tracking algorithm as part of a surveillance system in order to alert from the presence of any moving objects, in particular birds. It would therefore increase the safety in airports.
<p style="text-align: justify;">
This document describes the different components of this project and their usage. The provided technical work contains the following files and folders:

- ByteTrack: contains the tracking algorithm developed during this project
- Guide_Notebooks: contains two notebooks. One is a demonstration of the tracker's usage and the second one shows how to train a YOLO model with the custom Dataset
- IRP_Dataset: contains the custom dataset created during this project
- siammot: contains the source code for the SiamMOT model. This model is not correctly working
- Tracking_Outputs: contains the videos resulting from tracking on the testing set.


## <div align="center">Installation Instructions</div>
<p style="text-align: justify;">
The tracking algorithm has been developed using multiple Python libraries. In order for the tracker to work correctly, make sure the needed packages are installed on your working environment.

Pip install the required packages in a [**Python>=3.8**](https://www.python.org/) environment.

```bash
pip install -r requirements.txt
```

This project is mainly based on [**Ultralytics**](https://github.com/ultralytics/ultralytics) and **OpenCV** packages.

## <div align="center">Dataset</div>
<p style="text-align: justify;">
The custom dataset created for this project is provided in the <strong>IRP_Dataset</strong> ZIP file. In order to be used, it should be first unzipped. The dataset folder is organised as follows:

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
<p style="text-align: justify;">
It contains 4 different subfolders used for different tasks. The subfolders <strong>train</strong>, <strong>YOLO_train</strong>, and <strong>val</strong> contain two subsubfolders <strong>images</strong> and <strong>labels</strong>. These two subsubfolders contain folders for each different frame sequence with the frames and the labels files respectively. 
<p style="text-align: justify;">
The <strong>YOLO_train</strong> subfolder contains the data used for the training of the <a href="https://github.com/ultralytics/ultralytics"><strong>Ultralytics</strong></a> Yolov8x model. It contains two subfolders for the images and the label files respectively. The different frames extracted from different videos are not separated into different folder since the YOLOv8 model is a detection model that does not need to process each frame sequence individually. The label files are in the YOLO format and therefore cannot be used directly to train or validate a tracking model.
<p style="text-align: justify;">
The <strong>train</strong> subfolder has not been used directly in the project. It contains the same data save in the <strong>YOLO_train</strong> but the frames are stored in different folder in order to distinguish each frame sequence. In addition, the labels files are also in YOLO format, with an additional value corresponding to the tracking ID. This subfolder could be used to train directly a tracking model such as <a href="https://arxiv.org/abs/2105.11595"><strong>SiamMOT</strong></a>. It is important to note that the frame sequences in this subfolder do not correspond to videos recorded from a static camera position.
<p style="text-align: justify;">
The <strong>val</strong> subfolder contains the videos that have been used for the validation of the proposed tracking model. It contains 5 different frame sequences annotated for tracking in the YOLO format. 
<p style="text-align: justify;">
The <strong>test</strong> subfolder contains different frame sequences used to visually assess the tracking performance.
<p style="text-align: justify;">
  In addition, each folder containing a single frame sequence also contains a <code>meta_info.txt</code> file providing informations about the video.

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
### YOLOv8 + MotionDetector + Byte: ByteTrack
<p style="text-align: justify;">
The source code for the tracking system developed during this project is provided in the <strong>ByteTrack</strong> folder. This tracker is based on <a href="https://github.com/ultralytics/ultralytics"><strong>Ultralytics</strong></a> Yolov8 model, <strong>OpenCV</strong> background subtraction algorithm, and <a href="https://arxiv.org/abs/2110.06864"><strong>BYTE</strong></a> association algorithm. This tracking algorithm can be used as follows:

```python
import os
from glob import glob 
import cv2

from ByteTrack import Tracker # Import the Tracking model

# Load a model
model = Tracker("ByteTrack/weights/trained_yolov8x.pt")

# Use the model
metrics = model.val("IRP_Dataset/val")  # evaluate model performance on the validation set
results = model("IRP_Dataset/test/Test_01", fps=25)  # predict and track on an entire video

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
<p style="text-align: justify;">
There are multiple parameters that can be tuned depending on the situation. There are different possibilities to save the tracking results and change the detection model configuration. See the code documentation in <code>ByteTrack/engine/model.py</code> for more information.
<p style="text-align: justify;">
In addition, the configuration of the BYTE association algorithm is available in <code>ByteTrack/tracker/cfg</code>. There are two configuration files: one for the variation of the BYTE algorithm used in this project and a second one for the BoT-SORT algorithm based on the first association algorithm. This second matching method has not been tested an dused during this project but could be useful for further development of the work. These two configuration files contain the default values set for the different hyper-parameters important for tracking. By default, the Byte algorithm is used. The user can change the association algorithm by specifying the value of the argument <code>tracker</code>.

Refer to the Demonstration Notebook for an example of the tracking pipeline.


### SiamMOT
<p style="text-align: justify;">
In addition, the <strong>siammot</strong> folder contains the source code for a second tracking model <a href="https://arxiv.org/abs/2105.11595"><strong>SiamMOT</strong></a>. This model did not end up working but the code could be useful in some way for further works. As of now, only the training phase of the model has been implemented. In order to train the model, run the following command:

```
python3 siammot/train.py --config-file "siammot/configs/default.yaml" --source "IRP_Dataset/train" --epochs 20 --batch_size 1
```

<p style="text-align: justify;">
The model has been tested on a Crescent 2 account on a single GPU. The submission script is available in the folder <code>siammot</code>. Every python libraries required for this model are listed in the file <code>requirements.txt</code>. The model encountered some issues during training. First, the model is quite large and takes a lot of the GPU memory. In addition, the gradient computed during training and used to optimise the model's parameter happen to reach infinite values quickly. Finally, even clipping the gradient did not resolve the issue. Indeed, with a limited gradient, the different training losses did not decrease over the epochs.

## <div align="center">Tracking Outputs</div>
<p style="text-align: justify;">
The videos resulting from tracking on the testing set are available in the folder <strong>Tracking_Outputs</strong>. These videos visually show the tracking performance of the current tracking algorithm and could be used for comparison.

![frame1](https://github.com/theolange01/IRP/assets/116893751/e573920c-863b-4bfa-a78a-bf2432a9c940)
