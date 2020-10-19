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

int main(int argc, char *argv[])
{
  cv::Mat img1 = cv::imread(std::string(argv[1]));
  
  if(!img1.data)
  {
    cout<<"\nCould not load image from "<<argv[1]<<"\nExiting...";
    exit(0);
  }
  
   
  cv::Mat img2 = cv::imread(std::string(argv[2]));
  
  if(!img2.data)
  {
    cout<<"\nCould not load image from "<<argv[2]<<"\nExiting...";
    exit(0);
  }
  
  int diff = 0;
  int m = (int)img1.rows;
  int n = (int)img1.cols;
  int channels = (int)img1.channels();
  int cnt = 0;
  
  for(int i = 0; i < m; ++i)
  {
    for(int j = 0; j < n; ++j)
    {
      for(int k = 0; k < channels; ++k )
      {
        diff = img1.at<Vec3b>(i, j)[k] - img2.at<Vec3b>(i, j)[k];
        if(diff != 0)
        {
          ++cnt;
        }
      }
    }
  } 
  
  cout<<"\nNumber of differences = "<<cnt; 
  return 0;
}
