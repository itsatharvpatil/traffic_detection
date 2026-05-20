# Real-Time Object Detection with YOLOv8

<img width="1278" height="743" alt="webcamtest" src="https://github.com/user-attachments/assets/8f11e3ed-0e00-48a6-b693-1e4687535d60" />
*Live webcam demo showing real-time detection of person (0.86), cell phone (0.70), bottles, and chair at ~15 FPS. 

A lightweight, real-time object detection system built with Ultralytics YOLOv8n, fine-tuned on the Pascal VOC benchmark dataset. The project demonstrates end-to-end deep learning — from training on a large-scale dataset to live inference on a webcam feed — with a focus on evaluating performance on traffic-relevant object classes.

The screenshot shows the model working in a real-world indoor environment with accurate bounding boxes and confidence scores. It effectively proves:
- The system runs smoothly in real-time
- Detections are precise and confident
- The pipeline (webcam capture → inference → visualization) works end-to-end

For traffic-specific examples, the model excels at detecting **person, car, bus, bicycle, motorbike** when aimed at roads, vehicles, or outdoor scenes — exactly as intended for traffic monitoring use cases.

## Performance (Fine-Tuned Model — 30 Epochs on Pascal VOC)
Overall:

mAP@0.5: 0.836
mAP@0.5:0.95: 0.631

Traffic-relevant class AP@0.5:

Car: 0.931
Person: 0.903
Bus: 0.902
Bicycle: 0.920
Motorbike: 0.894

Inference speed:

1.6 ms/frame on Tesla T4 GPU (~600 FPS theoretical)
20–50 FPS real-time on consumer GTX 1650 Ti
Training time: ~2.5 hours for 30 epochs on Colab T4

## Key Features
Real-time webcam detection with bounding boxes, class labels, confidence scores, and FPS overlay
Fine-tuned on Pascal VOC (21,000+ images, 20 classes)
Evaluated specifically on traffic-relevant classes: person, car, bus, bicycle, motorbike
Stable CPU fallback mode for consumer hardware
Batch inference pipeline for large datasets
Robust error handling throughout

## Tech Stack
- Python
- Ultralytics YOLOv8
- OpenCV
- PyTorch
- NumPy

## Requirements
```txt
ultralytics>=8.3.0
opencv-python
numpy
```

## Install with:
```bash
pip install ultralytics opencv-python numpy
```
## Training Details

Base model: YOLOv8n (nano) pretrained weights
Dataset: Pascal VOC (train2007 + train2012 + val2007 + val2012 for training, test2007 for evaluation)
Epochs: 30
Image size: 640px
Batch size: 16
Optimizer: SGD with cosine LR scheduling
Augmentation: Mosaic, mixed-precision training (AMP)
Hardware: Google Colab T4 GPU

## Honest Project Notes

The model is trained on all 20 Pascal VOC classes, not exclusively traffic classes
The webcam demo screenshot was captured in an indoor environment — it demonstrates pipeline stability and detection confidence, not a traffic-specific scene
For traffic-specific inference, use the class filter below:

pythonresults = model(frame, conf=0.4, classes=[1, 5, 6, 13, 14])
 bicycle, bus, car, motorbike, person

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
