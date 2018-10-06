# CS131: Computer Vision Foundations and Applications
## 课程和作业介绍
This repository contains the released assignments for the fall 2017 iteration of [CS 131](http://vision.stanford.edu/teaching/cs131_fall1718/), a course at Stanford taught by [Juan Carlos Niebles](http://www.niebles.net) and [Ranjay Krishna](http://ranjaykrishna.com).

The assignments cover a wide range of topics in computer vision and should expose students to a broad range of concepts and applications. 

## 编程环境
- Python 3
- Jupyter Notebook
- Package 参照每个每份作业文件夹下的`requirement.txt`

提示：运行`pip install -r requirements.txt` 安装共享环境。

## Homework 0：Basics 

- 使用Python和Numpy操作图像
- 基础线性代数知识


## Homework 1：Filters - Instagram

- 理解基本概念
    - 卷积
    - 线性系统
    - 卷积核分解
- 设计卷积核来寻找图像的特定信号

## Homework 2: Edges - Smart Car lane Detection

- 边缘检测
- 霍夫变换检测直线
- 车道线检测

## Homework 3: Panorama - Image Stiching

- 介绍HOG和RANSAC
- 多幅图像中寻找匹配点
- 估计图像间仿射变换矩阵
- 实现拼接操作

## Homework 4: Resizing - Seam Carving

- 介绍seam carving算法
- 实现算法 
    - 定义图像能量
    - 动态编程寻找最小能量线
- 拓展实现图像缩放、目标移除

## Homework 5: Segmentation - Clustering

- 实现聚类算法
    - K-Means
    - HAC
- 提取图像特征序列进行分割
- 基于Groundtruth对分割算法进行量化评估

## Homework 6: Recognition - Classification

- SVD 图像压缩
- KNN 图像分类
- PCA(主成分分析)和LDA（线性判别分析）进行数据降维
 
 
## Homework 7: Object detection - constellation models

- Hog人脸表征
- 滑窗法人脸检测
- 图像金字塔解决维度诅咒
- DPM人脸检测

Homework 8 ends the course by moving away from still images and into video and studying time dependent features. They learn to use optical flow to calculate motion in images, a technique useful for tracking people in images and classifying actions that people perform.
