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
 * Calculates the Mean Squared Error between the corresponding plain images of 2 encrypted images, one of which is noisy
 */

static inline void MSE(cv::Mat img1,cv::Mat img2,int m,int n,int ch) 
{
  double init = 0,sum_mse = 0;
  
  for(int i = 0; i < m; ++i)
  {
    for(int j = 0; j < n; ++j)
    {
      for(int k = 0; k < ch; ++k)
      {
        
        init = (double)img1.at<Vec3b>(i,j)[k] - img2.at<Vec3b>(i,j)[k];
        sum_mse += fabs(init);
        
      }
    }
  }
  
  sum_mse = (sum_mse * 100) / (m * n * ch);
  printf("\nMSE = %F",sum_mse);
  
}

int main(int argc,char *argv[])
{
  std::string image_name_1 = std::string(argv[1]);
  std::string image_name_2 = std::string(argv[2]);
  
  cv::Mat img1 = cv::imread(image_name_1,cv::IMREAD_UNCHANGED);
  cv::Mat img2 = cv::imread(image_name_2,cv::IMREAD_UNCHANGED);
  
  int m = img1.rows;
  int n = img1.cols;
  int ch = img1.channels();
  
  cout<<"\nRows =  "<<m;
  cout<<"\nColumns = "<<n;
  cout<<"\nChannels = "<<ch;
  
  MSE(img1,img2,m,n,ch);
   
  return 0;
}
