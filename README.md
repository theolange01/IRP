# <div align="center">IRP - Development of a Tracking System for the Surveillance of Birds in Airports</div>

With the recent development of air travel as a common mean of transport, the question of safety and security in airports has become crucial. Amongst all the different existing danger, birds are considered as one of the main threats to aircraft safety. Bird strikes occur mainly during landing and take off. However, the current surveillance systems are not robust and accurate enough to alert from the presence of birds in restricted areas.

The main purpose of this IRP project is the development of a tracking solution for the surveillance of birds in airports to tackle the problem of birds monitoring. This tracking system aim to track and monitor birds in particular in restricted airports area in order to reduce the risks of bird strikes. The system developed during this project has beed designed to meet the following objectives:

- Track and monitor birds from a static camera position
- Follow the trajectory of any other moving objects
- Robustly handle challenging situations.

The tracking system proposed during this IRP aim to address the problem of birds monitoring in airports. The objective is to include the tracking algorithm as part of a surveillance system in order to alert from the presence of any moving objects, in particular birds. It would therefore increase the safety in airports.

This document describes the different components of this project and their usage.

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

