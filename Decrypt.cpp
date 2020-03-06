#include "functions.hpp"
using namespace std;
using namespace cv;

int main()
{

    // Read the file and confirm it's been opened
    Mat image= cv::imread("airplane_encrypted.png", IMREAD_COLOR);
    if (!image.data)
    {
        cout << "Image not found!\n";
        return -1;
    }
     
    if(RESIZE_TO_DEBUG==1)
    {
      cv::resize(image,image,cv::Size(2048,2048));
      
    }
    
    uint32_t m=0,n=0,total=0,cnt=0,cnt_file=0;
    uint32_t alpha=0,tme_8=0,manip_sys_time=0,element=0;
    uint64_t tme=0;
    uint16_t middle_element=0,xor_position=0;
    
    // Read image dimensions
    m = (uint32_t)image.rows; 
    n = (uint32_t)image.cols;
    total=m*n;
    uint32_t channels=(uint32_t)image.channels();  
    
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
    
    /*Read seed and system time value from file*/
    
    std::ifstream file("parameters.txt");
    if(!file)
    {
      cout<<"\nCould not open file "<<"parameters.txt\nExiting...";
      exit(0);
    }
    
    while (file >> element)
    {
      if(cnt_file==0)
      {
        alpha=element;
      }
      
      if(cnt_file==1)
      {
        manip_sys_time=element;
      }
      ++cnt_file;
    }
   
    file.close();
    
    printf("\nalpha in Decrypt= %d",alpha);
    printf("\nmanip_sys_time in Decrypt= %d",manip_sys_time);
    
    /*Generate PRNG*/
    generatePRNG(random_array,alpha,manip_sys_time);
    
    
    /*if(DEBUG_VECTORS==1)
    {
      printf("\nrandom_array in Decrypt=");

      for(uint32_t i=0;i<256;++i)
      {
        printf("%d ",random_array[i]);
      }
    }*/

    //Flattening Image
    flattenImage(image,img_vec);

    if(DEBUG_VECTORS==1)
    {
    
      cout<<"\nOriginal Image Vector in Decrypt=";
      for(uint32_t i=0;i<total*3;++i)
      {
        printf("%d ",img_vec[i]);
      }
    
      //printVectorCircular(img_vec,xor_position,total);
    
    }

    
    /*Doing prngStepOne
    prngStepOne(img_vec,random_array,total);
    
    if(DEBUG_VECTORS==1)
    {
      cout<<"\n\nimg_vec after prngStepOne in Decrypt=";
      for(int i=0;i<total*3;++i)
      {
        printf(" %d",img_vec[i]);
      }
      
      cout<<"\n\nrandom_array after prngStepOne in Decrypt=";
      for(int i=0;i<256;++i)
      {
        printf(" %d",random_array[i]);
      }
      std::string parameters =  std::string("");
      
      ofstream file("img_vec_dec.txt");
      if(! file)
      {
        cout<<"\nCould not find "<<"img_vec_enc.txt\n Exiting...";
        exit(0);
      }
      
      for(int i=0;i<total*3;++i)
      {
        parameters.append(std::to_string(img_vec[i]));
        parameters.append("\n");
      }
      file<<parameters;
      file.close();
    }*/

    //Obtaining middle element and xor position
    middle_element=0;
    xor_position=0;
    
    
    //Xoring image vector
    xorImageDec(img_vec,img_xor_vec,m,n);
    if(DEBUG_VECTORS==1)
    {    
      cout<<"\nXor'd Image Vector after prngStepOne in Decrypt=";
      for(int i=0;i<total*3;++i)
      {
        printf(" %d",img_vec[i]);
      }
    }
    
    if(DEBUG_IMAGES==1)
    {
      for(int i=0;i<total*3;++i)
      {
        img_arr[i]=img_xor_vec[i];
      }

      /*cout<<"\nimg_arr=";
      for(int i=0;i<total*3;++i)
      {
        printf(" %d",img_arr[i]);
      }*/
      cv::Mat img_reshape(m,n,CV_8UC3,img_arr);
      //imshow("image", cvImage);
      //waitKey(30);
      cv::imwrite("airplane_decrypted.png",img_reshape);
    }
    return 0;
}


