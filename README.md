# BasicCNN

### Introduction

This project delves into Convolutional Neural Networks (CNNs) for image classification, emphasizing hands-on learning. We implement a basic CNN architecture using TensorFlow, focusing on architecture design, data generation, and result assessment. Through model training and visualization, we explore fundamental CNN concepts applicable to various image classification tasks. By the project's end, we aim to provide a foundational understanding of CNNs, empowering further exploration in computer vision and deep learning.

### Model Architecture

![CNN architecture.](https://github.com/shubham031194/BasicCNN/blob/main/Model.png)

### About Dataset
![Dataset Sample](https://github.com/shubham031194/BasicCNN/blob/main/dataset.png)
The Sign Language MNIST dataset contains 27,455 grayscale images of 28x28 pixels, representing 24 classes of American Sign Language letters (excluding J and Z, which involve motion). It's adapted from Sign Language MNIST, converting CSV files to JPEG images. Each image is labeled with a corresponding letter (0-25). The dataset is organized in folders named after the class of images, compatible with TensorFlow data flow generators. Images undergo cropping, resizing, grayscale conversion, and various modifications, creating diverse training data. This augmentation strategy enhances class separation and resolution for improved model training.
[Dataset download page](https://www.kaggle.com/datasets/ash2703/handsignimages/data)

### Install packages
```
pip install tensorflow
pip install matplotlib
pip install numpy
pip install pandas
pip install opencv-python
pip install scikit-learn
```

### Project execution flow

Data Loading and Preprocessing --> Data Splitting --> Model Construction --> Model Training --> Model Evaluation --> Model Testing --> Result Analysis and Interpretation
