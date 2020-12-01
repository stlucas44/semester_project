#include "mesh_visualizer/mesh_visualizer.h"
#include "ros/ros.h"


meshVisualizer::meshVisualizer(ros::NodeHandle nh) {
  nh_ = nh;
  }

void meshVisualizer::setBool(){
  testBool = true;
  ROS_ERROR("Set bool worked.");
  }
