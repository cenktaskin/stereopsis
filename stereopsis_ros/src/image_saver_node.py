#! /usr/bin/python3

import rospy
from sensor_msgs import msg
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import cv2
import pytz
from datetime import datetime
from pathlib import Path


class ImageSaverNode:
    stereo_topic = '/stereo_driver/image_raw'
    ir_topic = '/pico_flexx/image_mono16'
    ir_topic8 = '/pico_flexx/image_mono8'
    depth_topic = '/pico_flexx/image_depth'
    ir_format = '16UC1'
    depth_format = '32FC1'

    def __init__(self, calibration, output_dir):
        timestamp = datetime.now().astimezone(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d%H%M")
        self.output_path = Path(output_dir).joinpath(f"dataset-{timestamp}")
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        self.bridge = CvBridge()
        self.image_count = 0
        self.calibration = calibration
        self.st_sub = message_filters.Subscriber(ImageSaverNode.stereo_topic, msg.Image)

        pico_topic = ImageSaverNode.depth_topic
        self.pico_format = ImageSaverNode.depth_format
        self.pico_prefix = 'dp_'
        if calibration:
            pico_topic = ImageSaverNode.ir_topic
            self.pico_format = ImageSaverNode.ir_format
            self.pico_prefix = 'ir_'

        self.pico_sub = message_filters.Subscriber(pico_topic, msg.Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.st_sub, self.pico_sub], 10, 0.1)
        ts.registerCallback(self.callback_poly)
        print("Node initialized")
        print(f"Saving images to -> {self.output_path}")

    def callback_poly(self, stereo_msg, pico_msg):
        try:
            stereo_img = self.bridge.imgmsg_to_cv2(stereo_msg, "bgr8")
            pico_img = self.bridge.imgmsg_to_cv2(pico_msg, self.pico_format)
        except CvBridgeError as e:
            print(e)
        else:
            time_stamp = stereo_msg.header.stamp
            cv2.imwrite(str(self.output_path.joinpath('st_' + str(time_stamp) + '.tiff')), stereo_img)
            cv2.imwrite(str(self.output_path.joinpath(self.pico_prefix + str(time_stamp) + '.tiff')), pico_img)
            self.image_count += 1
            print(f"Saved {self.image_count} image(s)")
            rospy.sleep(0.01)


if __name__ == '__main__':
    rospy.init_node('image_saver')
    ImageSaverNode(calibration=rospy.get_param('calibration', False),
                   output_dir=rospy.get_param('output_dir', Path.home().joinpath('/ros-outputs/')))
    while not rospy.is_shutdown():
        rospy.spin()
    print("Terminated")
