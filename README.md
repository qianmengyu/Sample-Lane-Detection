# Sample-Lane-Detection
A sample Lane detection using Digital image processing


## Introduction

code\VideoImageExtract：从视频中取出图片，100取1

code\PointPosition：鼠标取点

code\LaneLine1：1.1：Sobel阈值分割

code\LaneLine2：1.2：RGB阈值分割

code\LaneLine3：2.1：RGB阈值分割+三角区域

code\LaneLine4：2.2：Sobel阈值分割+梯形区域

code\LaneLine5：3.1：Sobel阈值分割+梯形区域+Hough变换

code\OutputVideo：将处理结果输出为视频

## yolo3 Detection

参考：
https://github.com/bubbliiiing/yolo3-pytorch

将里面的predict.py换为code中的predict.py


## Result

详情见out1和out2中视频。

白天：  
![alt text](other/img1_24.jpg)

黑夜：  
![alt text](other/img5_9.jpg)

白天行人车辆检测：  
![alt text](other/Picture1.png)

黑夜行人车辆检测：  
![alt text](other/Picture2.png)

