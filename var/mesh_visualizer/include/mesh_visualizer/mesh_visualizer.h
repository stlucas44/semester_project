#include "ros/ros.h"

class meshVisualizer {
  public:
    meshVisualizer(ros::NodeHandle nh);
    bool testBool;
    void setBool();

  private:
    int test;
    ros::NodeHandle nh_;
    //ros::NodeHandle nh_private_;

};
