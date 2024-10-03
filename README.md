# Computer Vision - ETH Zurich (2023-2024)

This repository hosts the **lab projects** of the Computer Vision course held by ETH Zurich during the Winter 2023.


# Lab Projects

## Lab 2: Feature extraction and Matching

In the first part of the lab exercise, we practice with the Harris corner detection algorithm in two different
input images. In the second part of the lab exercise, having created the basis on feature extraction with the Harris
detector, we experiment with several image matching techniques.

## Lab 3: Classification task with Pytorch

In this lab, several pytorch neural network architectures (perceptron and CNNs) are implemented for image classification tasks (MNIST dataset; classification on hand-written image of 10 digits, CIFAR10 dataset; classification of 60000 32 × 32 color images in 10 classes, with 6k images per class).

## Lab 4: Object Recognition

In the first part of the lab exercise, we implement a Bag of Visual Words Classifier that determines
whether an input image contains a car (label 1) or not (label 0). For the second task, we implement a simplified version of the VGG image classification network on
CIFAR-10 dataset.

## Lab 5: Image Segmentation

In the first part of the lab exercise, we implement the Mean-Shift algorithm for segmentation of an
input image, that illustrates ETH, in the CIELAB color space (downscaled by a factor of 0.5 for faster
computations). In the second part, we implement and train a simplified version of SegNet.

## Lab 6: Condensation Tracker

In this lab exercise, we implement a CONDENSATION tracker based on color histograms
using python. In this exercise we focus on tracking just one object and consider just two prediction
models (modeled by a matrix A): (i) no motion at all i.e. just noise; and (ii) constant velocity motion model.

## Lab 7: Structure from Motion & Model Fitting

In this assignment, we produce a reconstruction of a small scene using Structure from Motion
(SfM) methods. We combine realtive pose estimation, absolute pose estimation and point triangulation
to a simplified Structure from Motion pipeline. In the second part of this assignment, we implement RANSAC (RANdom SAmple Consensus) for robust model
fitting.
