#! /usr/bin/python3
import rospy
from sensor_msgs.msg import CameraInfo
import camera_info_manager


def info_callback(msg):
    camera_info_manager.saveCalibration(msg, "file:///home/cenkt/git/stereopsis/data/processed/pico_intrinsics.yaml",
                                        "pico_flexx")
    rospy.signal_shutdown("Saved the intrinsics")


if __name__ == '__main__':
    info_topic = '/pico_flexx/camera_info'
    rospy.init_node('cam_info_listener')
    info_sub = rospy.Subscriber(info_topic, CameraInfo, info_callback)
    while not rospy.is_shutdown():
        rospy.spin()
