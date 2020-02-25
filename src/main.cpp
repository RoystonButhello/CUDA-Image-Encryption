  #include <iostream> /*For IO*/
  #include <cstdint>  /*For standardized data types*/
  #include "functions.hpp"
  
  using namespace std;
  using namespace cv;
  int main()
  { 
    
    uint16_t m=0,n=0,total=0;

    
    cv::Mat image;
    
    image=cv::imread("lena.png",cv::IMREAD_COLOR);
    cv::resize(image,image,cv::Size(),0.01,0.01);
    
    m=(uint16_t)image.rows;
    n=(uint16_t)image.cols;
    
    cout<<"\nm= "<<m<<"\tn= "<<n;
    total=(m*n);
    bool isNotDecimal=0;
    
    /*Declarations*/
    
    /*Miscellaneous Declarations*/
    std::vector<float> P1(total);
    std::vector<float> P2(total);
    std::vector<uint8_t> U(m);
    std::vector<uint8_t> V(m);
    std::vector<uint8_t> img_vec(total*3);
    
    genRelocVec(U,P1,"constants1.txt",m,n,SEED1);
    genRelocVec(V,P2,"constants2.txt",n,m,SEED2);
    
    cout<<"\nP1=";
    printFloatVector(P1);
    cout<<"\nU=";
    for(uint32_t i=0;i<m;++i)
    {
      printf("%d ",U[i]);
    }
   cout<<"\n";
    printFloatVector(P2);
    cout<<"\nV=";
    for(uint32_t i=0;i<m;++i)
    {
      printf("%d ",V[i]);
    }
    cout<<"\n";
    
    flattenGrayScaleImage(image,img_vec);
    
    //cout<<"\nimg_vec=";
    /*for(uint32_t i=0;i<m;++i)
    {
      cout<<img_vec[i]<<" ";
    }*/
    
    
    return 0; 
  }
  

