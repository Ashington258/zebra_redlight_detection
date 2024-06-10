#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def callback(data):
    rospy.loginfo("Received detection labels: %s", data.data)

def listener():
    rospy.init_node('detection_labels_listener', anonymous=True)
    rospy.Subscriber("/detection_labels", String, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
