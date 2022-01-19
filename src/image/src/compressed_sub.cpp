#include <iostream>
#include <fstream>

#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

cv::Mat image(2048, 2448, CV_8UC3);
cv::Mat pts_img(2048, 2448, CV_8UC3);
boost::shared_ptr<pcl::PointCloud<pcl::PointXYZI>> point_cloud(new pcl::PointCloud<pcl::PointXYZI>);

bool b_display = true;
double disp_scale = 5.0;
double scale = 0.0001745329252 * 2;

// Camera Intrinsic
double fx = 2.4032691757255579e+03;
double fy = 2.4121512925696502e+03;
double cx = 1.2166054602770371e+03;
double cy = 1.0082126790786290e+03;

double k1 = -1.7891839184119299e-01;
double k2 = 1.5709201128078301e-01;
double p1 = 7.5643992933044482e-05;
double p2 = -2.0919911715141070e-04;
double k3 = 0.0;

// Rotation for Lidar to Sensor
double L2S_roll = 0.0;
double L2S_pitch = 0.0;
double L2S_yaw = 0.00872665;
//double L2S_yaw = 0.00872664626;

// Rotation for Sensor to Camera
double S2C_roll = -0.00789906;
double S2C_pitch = -0.0736555;
double S2C_yaw = 3.32434e-05;

// Translation for Sensor to Camera
double S2C_tx = 1.2810379227752366e-01;
double S2C_ty = -9.2835991418070245e-03;
double S2C_tz = -1.1765842063953642e-01;

cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
cv::Mat rvec = cv::Mat::eye(3, 3, CV_64FC1);
cv::Mat r_optical2sensor = cv::Mat::eye(3, 1, CV_64FC1);
cv::Mat t_sensor2optical = cv::Mat::eye(3, 3, CV_64FC1);

// Transform Camera to Optical [-pi/2, 0, -pi/2]
cv::Mat r_cam2optical = (cv::Mat_<double>(3, 3) << 0., 0., 1.,
                -1., 0., 0.,
                0., -1., 0.);

void updateMatrix()
{
    Eigen::AngleAxisf yawAngle(S2C_yaw, Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf pitchAngle(S2C_pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rollAngle(S2C_roll, Eigen::Vector3f::UnitX());

    // Validation :: Matlab eul2rotm
    Eigen::Quaternion<float> quatenion = yawAngle * pitchAngle * rollAngle;

    cv::Mat r_sensor2cam;
    cv::eigen2cv(quatenion.matrix(), r_sensor2cam);
    r_sensor2cam.convertTo(r_sensor2cam, CV_64F);

    // Translation for Lidar to Camera
    cv::Mat_<double> t_sensor2cam(3, 1); t_sensor2cam << S2C_tx, S2C_ty, S2C_tz;

    cv::Mat r_sensor2optical = r_sensor2cam * r_cam2optical;
    t_sensor2optical = r_cam2optical * t_sensor2cam;
    //std::cout << "r_sensor2optical = " << std::endl << r_sensor2optical << std::endl;
    //std::cout << "t_sensor2optical = " << std::endl  <<  t_sensor2optical << std::endl;

    r_optical2sensor = r_sensor2optical.t();
    tvec = -r_sensor2optical * t_sensor2optical;
    //std::cout << "r_optical2sensor = " << std::endl  <<  r_optical2sensor << std::endl;
    //std::cout << "t_optical2sensor = " << std::endl  <<  t_optical2sensor << std::endl;

    cv::Rodrigues(r_optical2sensor, rvec);
}

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    try
    {
        // ROS_INFO("sub_img");
        image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
        pts_img = image.clone();
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("Could not convert to image!");
    }
}

void Callback_point_cloud(const sensor_msgs::PointCloud2::ConstPtr &point_cloud_msg)
{
    // ROS_INFO("Sub_points");
    pcl::fromROSMsg(*point_cloud_msg, *point_cloud);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    cv::namedWindow("view", CV_WINDOW_NORMAL);
    cv::resizeWindow("view", 2448 / 2, 2048 / 2);

//    ros::Subscriber sub_img = nh.subscribe("/sensing/camera/traffic_light/image_raw", 1, imageCallback);
    ros::Subscriber sub_img = nh.subscribe("/sensing/camera/traffic_light/image_raw", 1, imageCallback);
    ros::Subscriber sub_points = nh.subscribe("/points_raw", 1, Callback_point_cloud);

    Eigen::AngleAxisf yawAngle(L2S_yaw, Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf pitchAngle(L2S_pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rollAngle(L2S_roll, Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> quatenion = yawAngle * pitchAngle * rollAngle;
    cv::Mat r_lidar2sensor;
    cv::eigen2cv(quatenion.matrix(), r_lidar2sensor);
    r_lidar2sensor.convertTo(r_lidar2sensor, CV_64F);

    cv::Mat cameraMat = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                         0.0, fy, cy,
                         0.0, 0.0, 1.0);

    cv::Mat distCoeff = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);

    while (ros::ok())
    {
        ros::spinOnce();
        updateMatrix();
        std::vector<cv::Point3f> sensor_pts;
        std::vector<double> distance;
        int point_cloud_size = point_cloud->size();
        for (int idx = 0; idx < point_cloud_size; idx++)
        {
            cv::Mat lidar_mat = (cv::Mat_<double>(3, 1) << point_cloud->points[idx].x,
                                        point_cloud->points[idx].y,
                                        point_cloud->points[idx].z);

            cv::Mat sensor_mat = r_lidar2sensor * lidar_mat;
            cv::Mat optical_mat = r_optical2sensor * lidar_mat + t_sensor2optical;
            if (optical_mat.at<double>(2) <= 2.) continue;
            distance.push_back(optical_mat.at<double>(2));
            sensor_pts.push_back(cv::Point3f(sensor_mat.at<double>(0), sensor_mat.at<double>(1), sensor_mat.at<double>(2)));
        }

        std::vector<cv::Point2f> projected_img_pts;
        if (!sensor_pts.empty()) cv::projectPoints(sensor_pts, rvec, tvec, cameraMat, distCoeff, projected_img_pts);

        for (int idx = 0; idx < projected_img_pts.size(); idx++)
        {
            if (projected_img_pts[idx].x < 0. || projected_img_pts[idx].y < 0.) continue;
            if (projected_img_pts[idx].x >= 2448. || projected_img_pts[idx].y >= 2048.) continue;
            
            cv::circle(pts_img, projected_img_pts[idx], 2, CV_RGB(255 - (int)(disp_scale * distance[idx]), (int)(disp_scale * distance[idx]), 0), -1);
        }
    
        if(b_display)   cv::imshow("view", pts_img);
        else            cv::imshow("view", image);

        
        int chkey = cv::waitKey(1);
        if (chkey == 27) return -1;
        switch(chkey)
        {
            case 'q' : S2C_yaw += scale; break;
            case 'a' : S2C_yaw -= scale; break;
            case 'w' : S2C_pitch += scale; break;
            case 's' : S2C_pitch -= scale; break;
            case 'e' : S2C_roll += scale; break;
            case 'd' : S2C_roll -= scale; break;
            case 'o' : disp_scale += 0.25; break;
            case 'l' : disp_scale -= 0.25; break;
            case 'p' : b_display = !b_display; break;
            case 32 :
            std::cout << "roll: " << S2C_roll << std::endl;
            std::cout << "pitch: " << S2C_pitch << std::endl;
            std::cout << "yaw: " << S2C_yaw << std::endl;
            break;
        }
    }
}
