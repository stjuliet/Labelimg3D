# 3D bbox 标注工具使用说明

## 1、开发/运行环境

|          |                                                              |
| -------- | ------------------------------------------------------------ |
| 操作系统 | Windows 10 企业版                                            |
| 工具     | pycharm(2018.2 专业版)、Anaconda(5.3.1, python3.7/3.6)、PyQt5 (5.15) |
| 库       | OpenCV 4.4.0、YOLOv4(Darknet)                                |

## 2、操作说明

### 1. 打开软件主界面：

​	主界面包含四部分：菜单栏、图片显示区、标注选项区、文件列表区。

![main-page](https://github.com/stjuliet/Labelimg3D/blob/master/pictures/main-page.png)



### 2、选择标注文件夹：

​	点击菜单栏"Menu"，选择标注图片所在文件夹，软件自动读取文件夹下的所有图片文件至文件列表区。![1602038521283](https://github.com/stjuliet/Labelimg3D/blob/master/pictures/1602038521283.png)



### 3、选择标注文件：

​	双击文件列表区的任意一行，软件自动对图片中的车辆目标进行检测(只检测car\truck\bus)，并将图片中的车辆目标数显示于标注选项区。

![1602038951900](https://github.com/stjuliet/Labelimg3D/blob/master/pictures/1602038951900.png)



### 4、选择任意车辆对象进行标注：

​	（1）通过标注选项区的按钮切换选择想要标注的车辆目标，软件会自动获得目标检测类型，如果有误检测，可通过下拉框重新选择车辆类型；

​	（2）选择标注基准点(目标检测框左/右下角)；

​	（3）调节基准点、消失点及车辆物理尺寸使得3d bbox与二维检测框具有最高的贴合程度；

​	（4）当标注完成一个车辆时，使用"Ctrl+A"保存标注结果，如果标注有误，可点击"Clear"撤销重新进行标注，重复（1）--（4）完成一张图像中所有可标注车辆的标注；

​	（5）当一张图像中所有可标注车辆均已完成标注，使用"Ctrl+S"保存全部标注结果，输出xml标注文件至图像所在路径，完成标注；

​	（6）重复（1）--（5）完成标注文件夹中所有图片的标注。

当有标注错误时，可点击界面上新增的Clear按钮清除，然后重新标注。



标注示例：

xml文件包含：图片路径、图片尺寸/通道数、对应场景标定文件路径、车辆类型、2d box左上角点图像坐标及像素宽高、3d bbox在图像中对应8个顶点坐标、车辆三维物理尺寸（单位：m）、基准点方向（视角）、基准点图像坐标、3d bbox在真实世界中对应8个顶点坐标、车辆3d bbox中心点图像坐标。

![1602039817322](https://github.com/stjuliet/Labelimg3D/blob/master/pictures/1602039817322.png)

![example-annotation](https://github.com/stjuliet/Labelimg3D/blob/master/pictures/example-annotation.png)

8个点坐标存储顺序：

|                          基准点-左                           |                          基准点-右                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![1602047236054](https://github.com/stjuliet/Labelimg3D/blob/master/pictures/1602047236054.png) | ![1602047244453](https://github.com/stjuliet/Labelimg3D/blob/master/pictures/1602047244453.png) |



## 附录

### 1、文件目录结构

针对每一个场景，单独建一个文件夹，同时放置标定文件。

(目录树制作参考：https://blog.csdn.net/qq_36910634/article/details/103888113)

├─model_yolov4  # 模型存放文件夹

│      coco.names  # 类型文件

│      yolov4.cfg  # 模型配置文件

│      yolov4.weights  # 模型权重

│      

├─test_images(支持jpg/png/bmp格式)  # 需要标注文件存放文件夹

│  │  session0_centre_scene5060.bmp

│  │  session0_centre_scene5061.bmp

│  │  session0_centre_scene5062.bmp

│  │  session0_centre_scene5063.bmp

│  │  session0_centre_scene5064.bmp

│  │  session0_centre_scene5065.bmp

│  │  session0_centre_scene5066.bmp

│  │  session0_centre_scene5067.bmp

│  │  session0_centre_scene5068.bmp

│  │  session0_centre_scene5069.bmp

│  │  session0_centre_scene5070.bmp

│  │  session0_centre_scene5071.bmp

│  │  session0_centre_scene5072.bmp

│  │  session0_centre_scene5073.bmp

│  │  session0_centre_scene5074.bmp

│  │  session0_centre_scene5075.bmp

│  │  session0_centre_scene5076.bmp

│  │  session0_centre_scene5077.bmp

│  │  session0_centre_scene5078.bmp

│  │  session0_centre_scene5079.bmp

│  │  session0_centre_scene5080.bmp

│  │  session0_centre_scene5081.bmp

│  │  session0_centre_scene5082.bmp

│  │  session0_centre_scene5083.bmp

│  │  session0_centre_scene5084.bmp

│  │  session0_centre_scene5085.bmp

│  │  session0_centre_scene5086.bmp

│  │  session0_centre_scene5087.bmp

│  │  session0_centre_scene5088.bmp

│  │  session0_centre_scene5089.bmp

│  │  session0_centre_scene5090.bmp

│  │  session0_centre_scene5091.bmp

│  │  session0_centre_scene5092.bmp

│  │  

│  └─calib   # 标定文件存放文件夹

│          session0_centre_calibParams.xml   # 标定文件



### 2、关于使用Pyinstaller打包exe遇到的问题

打包命令行：

`pyinstaller -F -w -i resource/labelimg3d.ico main.py`

- [x] 开发标注工具时，使用了带Cuda加速的OpenCV DNN模块调用最新的YOLOv4目标检测模型，在封装时出现以下问题：![1602041813513](https://github.com/stjuliet/Labelimg3D/blob/master/pictures/1602041813513.png)

  查阅了很多解决方案，都说重新安装，但是一旦重新安装，编译过的功能就会失效。

  考虑到只是对单张图片进行目标检测，而且软件中本身就使用了多线程，不会导致界面卡死，因此在anaconda中重新创建了一个env，在其中安装原始版本的OpenCV4.4.0用于目标检测。

  经过测试，在IDE中能够正常运行，使用pyinstaller打包exe也能够正常运行。



### 3、网络损失函数设计

- 网络输入：RGB图像
- 网络输出：车辆类型热力图、车辆中心点图像坐标、3d bbox的8个顶点图像坐标、车辆三维物理尺寸
- 车辆类别损失（focal loss）
- 8点二维图像坐标损失（L1）
- 车辆三维尺寸损失（L1）
- 由基准点反算至世界坐标，根据预测所得长宽高+标定参数推算8点二维图像坐标，再计算8点二维图像坐标损失（L1）
- 3d iou损失，将预测8点构成的3d box通过相机标定参数和预测的长宽高反算至世界坐标中，与真实标注的世界坐标中的3d box求3d iou确定loss，参考ciou loss的论文实现：Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression [[arxiv]](http://arxiv.org/abs/1911.0828)
