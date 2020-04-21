#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"

int main()
{
  cv::Mat image;
  cv::Mat img_decrypt;
  image = cv::imread("mountain1080.png",cv::IMREAD_COLOR);
  //cv::resize(image,image,cv::Size(1024,1024));
  cv::imwrite("mountain1080_1.png",image);
  img_decrypt = cv::imread("mountain1080_decrypted.png",cv::IMREAD_COLOR);
  uint32_t m = image.rows;
  uint32_t n = image.cols;
 int count_differences = 0;
  for(uint32_t i = 0; i < m; ++i)
  {
    for(uint32_t j = 0; j < n; ++j)
    {
      for(int k = 0; k < 3; ++k)
      {
        if(image.at<Vec3b>(i,j)[k] - img_decrypt.at<Vec3b>(i,j)[k] != 0)
        {
          ++count_differences;
        }
      }
    }
  }
  printf("\ncount_differences = %d",count_differences);
  return 0;
}
