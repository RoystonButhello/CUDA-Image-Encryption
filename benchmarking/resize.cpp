#include <iostream> /*For IO*/
#include <cstdlib>  /*For standard C functions*/
#include <cstring>  /*For char * strings */
#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp> /*For OpenCV*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
/**
 * Resize an image
 */
static inline std::string getFileNameFromPath(std::string filename)
{
  const size_t last_slash_idx = filename.find_last_of("\\/");
  if (std::string::npos != last_slash_idx)
  {
    filename.erase(0, last_slash_idx + 1);
  }

  // Remove extension if present.
  const size_t period_idx = filename.rfind('.');
  if (std::string::npos != period_idx)
  {
    filename.erase(period_idx);
  }
      
  return filename;
}

int main(int argc,char *argv[])
{
  std::string image_path = std::string(argv[1]);
  std::string image_name = getFileNameFromPath(image_path);

  int rows = atoi((const char*)argv[2]);
  int cols = atoi((const char*)argv[3]);
  cv::Mat image = cv::imread(image_path,cv::IMREAD_UNCHANGED);
  cv::resize(image,image,cv::Size(cols,rows),CV_INTER_LANCZOS4);
  std::string new_image_name = "";
  new_image_name  = new_image_name + image_name +  "_" + std::to_string(rows) + "_" + std::to_string(cols) + ".png";
  cout<<"\nNew image name = "<<new_image_name;
  cv::imwrite(new_image_name,image);
  return 0;
}
