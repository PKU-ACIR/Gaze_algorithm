# 说明文档

1. 模型文件路径
   - /object_tracking/deep_sort/deep/checkpoint/cpkt.t7
     - DeepSORT深度网络部分模型文件
   - /Detectron2/models/R101FPN.pkl
     - Detectron2实例分割模型文件
   - /models/eth-xgaze_resnet18.pth
     - ETH-XGaze视线检测模型文件
   - /models/rnn1.pt
     - 概率模型RNN模型文件
2. 模型文件下载链接
   - **待更新**
3. Dependencies
   - python ==  3.7
   - pytorch == 1.8.1+cu111(1.7以上)
   - pyrealsense2
   - open3d
   - Detectron2: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
   - yolact(可能需要根据环境自己编译DCNv2-latest)
   - omegaconf == 2.1.1
   - yacs == 0.1.8
   - dlib == 19.22.1
   - opencv-python == 4.5.3
   - 待更新

4. 工作路径说明
   - Detectron2用于储存与Detectron2库相关的设置与模型，此库本身需要自行安装至计算机中
   - object_tracking路径下为物体追踪的算法，yolact路径下为yolact实例分割方法，它们需要一起使用。
   - gaze_estimation路径下为实现识别的方法，里面还包括描述人脸各部位的类和简单的头部姿态检测方法。
5. 模型
   - 可以在百度网盘下载 链接：https://disk.pku.edu.cn/link/AAFEAEB524D3FC404989041B0F9F486140
