#include <iostream> /*For IO*/
#include <cstdlib>  /*For standard C functions*/
#include <cstring>  /*For char * strings */
#include <opencv2/opencv.hpp> /*For OpenCV*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/**
 * Replace a pixel in an image
 */

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
  
  std::string name = "lena";
  std::string image_name = name + ".png";
  std::string new_image_name = name + "_1pix_" + ".png";
  cv::Mat image = cv::imread(image_name,cv::IMREAD_UNCHANGED);
  cout<<"\n"<<image.rows;
  cout<<"\n"<<image.cols;
  cout<<"\n"<<image.channels();
  image.at<Vec3b>(0,0) = 0;
  cv::imwrite(new_image_name,image);
  cout<<"\n"<<new_image_name;
  return 0;
}
