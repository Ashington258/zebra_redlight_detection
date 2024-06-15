<!--
 * @Author: Ashington ashington258@proton.me
 * @Date: 2024-06-15 22:24:46
 * @LastEditors: Ashington ashington258@proton.me
 * @LastEditTime: 2024-06-15 22:33:24
 * @FilePath: \zebra_redlight_detection\zebra_redlight_detection_ai\doc\ROS消息格式.md
 * @Description: 请填写简介
 * 联系方式:921488837@qq.com
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved.
-->

ROS 消息结构，python 中使用 roslibpy

```python


def draw_bbox(bbox, names, F, H):
    det_result_str = ""
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4]) < float(0.05):
            continue
        bbox_height = bbox[idx][3] - bbox[idx][1]
        distance = (H * F) / bbox_height
        det_result_str += "{} {:.4f} distance: {:.2f} mm\n".format(names[int(class_id)], bbox[idx][4], distance)

    return det_result_str

```

ROS 系统使用 CPP 进行接收

```C
bool containsWord(const std::string& str,const std ::string& word)
{
  return  str.find(word)!=std::string::npos;
}

//判断是否包含关键字
void L1Controller::lightCB(const std_msgs::String::ConstPtr& msg)
{ //交通灯回调函数
    ROS_INFO("NB");
  if(containsWord(msg->data,"redlight"))
    {
      traffic_flag = false;
      ROS_INFO("redlight");
    }


    else if (msg->data == "zebra")
    {
      traffic_flag = false;
    }


    else
    {
      traffic_flag = true;
    }
}

```

现在需要增加距离判断，到达一定距离才对标志位进行切换

```CPP
bool containsWord(const std::string& str, const std::string& word)
{
  return str.find(word) != std::string::npos;
}

float extractDistance(const std::string& str)
{
  std::regex regex("distance: ([0-9.]+) mm");
  std::smatch match;
  if (std::regex_search(str, match, regex) && match.size() > 1)
  {
    return std::stof(match.str(1));
  }
  return -1.0f;  // Return -1 if no distance found
}

void L1Controller::lightCB(const std_msgs::String::ConstPtr& msg)
{ //交通灯回调函数
  ROS_INFO("NB");

  float distance = extractDistance(msg->data);
  if (distance > 0 && distance < THRESHOLD_DISTANCE) // Adjust the threshold distance as needed
  {
    if (containsWord(msg->data, "redlight"))
    {
      traffic_flag = false;
      ROS_INFO("redlight");
    }
    else if (msg->data == "zebra")
    {
      traffic_flag = false;
    }
    else
    {
      traffic_flag = true;
    }
  }
  else
  {
    ROS_INFO("Distance is greater than threshold or not found.");
    traffic_flag = true;
  }
}

```

增加两个距离宏定义
车灯的和斑马线的