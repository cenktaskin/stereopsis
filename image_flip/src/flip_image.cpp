#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

//I got this from add_image_flip branch if image_pipeline, and modified it a bit

class ImageFlipper{
private:
    ros::NodeHandle _nh;
    image_transport::ImageTransport _it;
    image_transport::Publisher _pub;
    image_transport::Subscriber _sub;
    sensor_msgs::ImagePtr _msg;
    cv::Mat _cvmat;
    int _flip_value;

public:
    ImageFlipper(int flip_value)
        :_it(_nh)
    {
        _flip_value = flip_value;
        _pub = _it.advertise("flipped", 1);
        _sub = _it.subscribe("image", 1, &ImageFlipper::image_cb, this);
        if (ros::names::remap("image") == "image") {
          ROS_WARN("Topic 'image' has not been remapped! Typical command-line usage:\n"
                   "\t$ rosrun image_flip image_flip image:=<image topic> flipped:=<flipped image topic> <horizontal/vertical/both>");
        }
        ROS_INFO_STREAM("Flipping "
                        <<  ( (_flip_value == 1)?"horizontally ":"" )
                        <<  ( (_flip_value == 0)?"vertically ":"" )
                        <<  ( (_flip_value == -1)?"both horizontally and vertically ":"" )
                        << "topic "
                        << _sub.getTopic() << " into " << _pub.getTopic());
    }

    void image_cb(const sensor_msgs::ImageConstPtr& incoming_img){
        // Do nothing if no one is subscribed
        if (_pub.getNumSubscribers() < 1)
            return;

        // ROS image message to opencv
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
          cv_ptr = cv_bridge::toCvCopy(incoming_img, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
          ROS_ERROR("cv_bridge exception: %s", e.what());
          return;
        }
        cv::flip(cv_ptr->image, _cvmat, _flip_value);

        _msg = cv_bridge::CvImage(incoming_img->header, "bgr8", _cvmat).toImageMsg();
        _pub.publish(_msg);
    }


};



int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_flip", ros::init_options::AnonymousName);
    int flip_value = -1;

    ImageFlipper imageflipper(flip_value);
    ros::spin();
    //    }
}
