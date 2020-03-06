#include "functions.hpp"
using namespace std;
using namespace cv;

int main()
{

    // Read the file and confirm it's been opened
    Mat image= cv::imread("airplane.png", IMREAD_COLOR);
    if (!image.data)
    {
        cout << "Image not found!\n";
        return -1;
    }
     
    if(RESIZE_TO_DEBUG==1)
    {
      cv::resize(image,image,cv::Size(2048,2048));
    }
    
    uint32_t m=0,n=0,cnt=0;
    uint32_t total=0;
    uint32_t alpha=0,tme_8=0,manip_sys_time=0;
    uint64_t tme=0;
    uint16_t middle_element=0,xor_position=0;

    
    
    // Read image dimensions
    m = (uint16_t)image.rows; 
    n = (uint16_t)image.cols;
    uint16_t channels=(uint16_t)image.channels();  
    total=m*n;
    
    /*Declarations*/
    uint8_t *img_arr=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
    std::vector<uint8_t> random_array(256);
    std::vector<uint8_t> img_vec(total*3);
    std::vector<uint8_t> img_xor_vec(total*3);
 

    cout<<"\nrows= "<<m;
    cout<<"\ncolumns="<<n;
    cout<<"\nchannels="<<channels;
    cout<<"\ntotal="<<total;
    
    /*Print the image matrix*/
    if(PRINT_IMAGES==1)
    {
      printImageContents(image); 
    }
   
    
   
    /*Generate seed and system time value*/
    alpha=getSeed(1,32);
    manip_sys_time=(uint32_t)getManipulatedSystemTime();
    
    printf("\nalpha= %d",alpha);
    printf("\nmanip_sys_time =%d",manip_sys_time);
    
    /*Write seed and system time value to file*/
    std::string parameters=std::string("");
    std::ofstream file("parameters.txt");
    
    if(!file)
    {
      cout<<"\nCould not open file "<<"parameters.txt"<<"\nExiting...";
      exit(0);
    }
    
    parameters.append(std::to_string(alpha));
    parameters.append("\n");
    parameters.append(std::to_string(manip_sys_time));
    parameters.append("\n");
    
    file<<parameters;
    file.close();
 
    /*Generate PRNG*/
    generatePRNG(random_array,alpha,manip_sys_time);
    
    
    /*if(DEBUG_VECTORS==1)
    {
      printf("\nrandom_array=");

      for(uint32_t i=0;i<256;++i)
      {
        printf("%d ",random_array[i]);
      }
    }*/

    //Flattening Image, obtaining middle element and xor position
    flattenImage(image,img_vec);
    
    /*if(DEBUG_VECTORS==1)
    {
    
      cout<<"\nOriginal Image Vector=";
      for(uint32_t i=0;i<total*3;++i)
      {
        printf("%d ",img_vec[i]);
      }
      std::string parameters=std::string("");
      ofstream file("img_vec_original.txt");
      if(!file)
      {
        cout<<"\n Could not open "<<"img_vec_original.txt";
        exit(0);
      }
      
      for(int i=0;i<total*3;++i)
      {
        parameters.append(std::to_string(img_vec[i]));
        parameters.append("\n");
      }
      file<<parameters;
      file.close();
      //printVectorCircular(img_vec,xor_position,total);
    
    }*/ 
     
    middle_element=0;
    xor_position=0;
    
    //printf("\nmiddle_element= %d",middle_element);
    //printf("\nxor_position= %d",xor_position); */    

    //Xoring image vector
    
    xorImageEnc(img_vec,img_xor_vec,m,n);
    
    if(DEBUG_VECTORS==1)
    {    
      cout<<"\nXor'd Image Vector before prngStepOne=";
      for(int i=0;i<total*3;++i)
      {
        printf(" %d",img_vec[i]);
      }
    }
  
    /*Doing prngStepOne
    prngStepOne(img_vec,random_array,total);
    
    if(DEBUG_VECTORS==1)
    {
      cout<<"\n\nimg_vec after prngStepOne=";
      for(int i=0;i<total*3;++i)
      {
        printf(" %d",img_vec[i]);
      }
      
      
      
      cout<<"\n\nrandom_array after prngStepOne=";
      for(int i=0;i<256;++i)
      {
        printf(" %d",random_array[i]);
      }
    }*/

    if(DEBUG_IMAGES==1)
    {
      for(int i=0;i<total*3;++i)
      {
        img_arr[i]=img_xor_vec[i];
      }
      
      cv::Mat img_reshape(m,n,CV_8UC3,img_arr);
      cv::imwrite("airplane_encrypted.png",img_reshape);
    }
    
    
    
    return 0;
}


