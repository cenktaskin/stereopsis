#! /usr/bin/python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import cv2
from pathlib import Path


class ImageSaverNode:
    output_file_path = Path('/home/cenkt/ros/stereopsis-ws/outputs/')

    def __init__(self, st_topic, ir_topic, dp_topic, ir_topic8, simult=True):
        self.bridge = CvBridge()
        self.image_count = 0

        if simult:
            st_sub = message_filters.Subscriber(st_topic, Image)
            ir_sub = message_filters.Subscriber(ir_topic, Image)
            dp_sub = message_filters.Subscriber(dp_topic, Image)
            ir_sub8 = message_filters.Subscriber(ir_topic8, Image)
            ts = message_filters.ApproximateTimeSynchronizer([st_sub, ir_sub, dp_sub, ir_sub8], 10, 0.01)
            ts.registerCallback(self.callback_poly)
        else:
            st_sub = rospy.Subscriber(st_topic, Image, self.callback_mono)

    def callback_mono(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            time = msg.header.stamp
            cv2.imwrite(str(ImageSaverNode.output_file_path.joinpath('st_' + str(time) + '.tiff')), cv2_img)
            self.image_count += 1
            print(f"Saved {self.image_count} image(s)")
            rospy.sleep(1)

    def callback_poly(self, stereo_msg, ir_msg, dp_msg, ir2_msg):
        try:
            stereo_img = self.bridge.imgmsg_to_cv2(stereo_msg, "bgr8")
            ir_img = self.bridge.imgmsg_to_cv2(ir_msg, "16UC1")
            ir_img8 = self.bridge.imgmsg_to_cv2(ir2_msg, "8UC1")
            depth_img = self.bridge.imgmsg_to_cv2(dp_msg, "32FC1")
        except CvBridgeError as e:
            print(e)
        else:
            time_stamp = stereo_msg.header.stamp
            cv2.imwrite(str(ImageSaverNode.output_file_path.joinpath('st_' + str(time_stamp) + '.tiff')), stereo_img)
            cv2.imwrite(str(ImageSaverNode.output_file_path.joinpath('ir_' + str(time_stamp) + '.tiff')), ir_img)
            cv2.imwrite(str(ImageSaverNode.output_file_path.joinpath('i8_' + str(time_stamp) + '.tiff')), ir_img8)
            cv2.imwrite(str(ImageSaverNode.output_file_path.joinpath('dp_' + str(time_stamp) + '.tiff')), depth_img)
            self.image_count += 1
            print(f"Saved {self.image_count} image(s)")
            rospy.sleep(1)


if __name__ == '__main__':
    stereo_topic = '/stereo_driver_node/image_raw'
    ir_topic = '/pico_flexx/image_mono16'
    ir_topic8 = '/pico_flexx/image_mono8'
    depth_topic = '/pico_flexx/image_depth'
    rospy.init_node('image_saver')
    print("Node initialized")
    ImageSaverNode(stereo_topic, ir_topic, depth_topic, ir_topic8)
    while not rospy.is_shutdown():
        rospy.spin()
    print("Terminated")
