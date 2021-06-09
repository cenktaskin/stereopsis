#! /usr/bin/python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters
from pathlib import Path
import sys


class ImageSaverNode():

    def __init__(self, st_topic, ir_topic, simult=True):
        self.bridge = CvBridge()
        self.image_count = 0

        if simult:
            st_sub = message_filters.Subscriber(st_topic, Image)
            ir_sub = message_filters.Subscriber(ir_topic, Image)
            ts = message_filters.ApproximateTimeSynchronizer([st_sub, ir_sub], 10, 0.1)
            ts.registerCallback(self.callback_duo)
        else:
            st_sub = rospy.Subscriber(st_topic, Image, self.callback_mono)

    def callback_mono(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            time = msg.header.stamp
            output_file_path = Path('/home/cenkt/projektarbeit/img_out/')
            cv2.imwrite(str(output_file_path.joinpath('st_' + str(time) + '.jpeg')), cv2_img)
            self.image_count += 1
            print(f"Saved {self.image_count} image(s)")
            rospy.sleep(1)

    def callback_duo(self, stereo_msg, ir_msg):
        try:
            stereo_img = self.bridge.imgmsg_to_cv2(stereo_msg, "bgr8")
            ir_img = self.bridge.imgmsg_to_cv2(ir_msg, "mono16")
        except CvBridgeError as e:
            print(e)
        else:
            time_stereo = stereo_msg.header.stamp
            time_ir = ir_msg.header.stamp
            output_file_path = Path('/home/cenkt/projektarbeit/img_out/')
            cv2.imwrite(str(output_file_path.joinpath('st_' + str(time_stereo) + '.jpeg')), stereo_img)
            cv2.imwrite(str(output_file_path.joinpath('ir_' + str(time_ir) + '.jpeg')), ir_img)
            self.image_count += 1
            print(f"Saved {self.image_count} image(s)")
            rospy.sleep(1)
            exit("EXIT")


if __name__ == '__main__':
    stereo_topic = '/stereo_driver_node/image_raw'
    ir_topic = '/pico_flexx/image_mono16'
    rospy.init_node('image_saver')
    print("Node initialized")
    ImageSaverNode(stereo_topic,ir_topic)
    while not rospy.is_shutdown():
        rospy.spin()
    print("Terminated")
