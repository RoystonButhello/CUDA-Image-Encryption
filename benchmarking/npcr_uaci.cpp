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

/*Calculates the Number of Pixels Change Rate between two encrypted images whose corresponding plain images differ by a single pixel*/

static inline void NPCR_UACI(cv::Mat img1,cv::Mat img2,int m,int n,int ch) 
{
  double NPCR = 0,UACI = 0,init = 0,sum = 0,sum_uaci = 0;
  
  for(int i = 0; i < img1.rows; ++i)
  {
    for(int j = 0; j < img1.cols; ++j)
    {
      for(int k = 0; k < img1.channels(); ++k)
      {
        
        init = (double)img1.at<Vec3b>(i,j)[k] - img2.at<Vec3b>(i,j)[k];
        sum_uaci += fabs(init);
        if(img1.at<Vec3b>(i,j)[k] != img2.at<Vec3b>(i,j)[k])
        {
          NPCR = 1;
          sum += NPCR;
        }
      }
    }
  }
  
  sum = (sum * 100) / (m * n * ch);
  printf("\nNPCR = %F",sum);
  sum_uaci = sum_uaci / (m * n * 255 * ch);
  printf("\nUACI = %F",sum_uaci * 100);
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
  
  NPCR_UACI(img1,img2,m,n,ch);
  
  return 0;
}
