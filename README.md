<!--
 * @Author: Ashington ashington258@proton.me
 * @Date: 2024-06-10 20:31:34
 * @LastEditors: Ashington ashington258@proton.me
 * @LastEditTime: 2024-06-10 22:17:23
 * @FilePath: \zebra_redlight_detection\README.md
 * @Description: 请填写简介
 * 联系方式:921488837@qq.com
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
-->
# 说明

该项目用于红绿灯识别和斑马线识别，
ai是使用人工智能的方案
cv是使用传统计算机视觉方案


## 关于docker

需要cv测试环境可以使用dockerfile但是似乎还有bug，慎用，ubantu可以直接使用setup.sh安装或者尝试自行构建环境，通过requirements

需要测试ai环境下的roslibpy，需要启动roscore，但是由于Atlas开发板镜像问题，无法兼容ROS1可以尝试docker部署

`docker run -p 9090:9090 -v /:/dockerfile -it --env="DISPLAY=$DISPLAY"  --name=rospylib ros:melodic  /bin/bash`
