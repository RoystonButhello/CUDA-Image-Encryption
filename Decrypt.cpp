#include "functions.hpp"
using namespace std;
using namespace cv;

int main()
{

    /*std::clock_t c_start = std::clock();
    // your_algorithm
    std::clock_t c_end = std::clock();

    long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";*/
    
    
    
    uint32_t m=0,n=0,cnt=0;
    uint32_t total=0;
    uint32_t alpha=0,tme_8=0,manip_sys_time=0;
    uint64_t tme=0;
    uint16_t middle_element=0,xor_position=0;
    long double time_array[18];
    long double total_time=0.00,average_time=0.00;
    
    // Read the file and confirm it's been opened
    
    std::clock_t img_read_start=std::clock();
    
    Mat image= cv::imread("airplane_encrypted.png", IMREAD_COLOR);
    std::clock_t img_read_end=std::clock();
    
    time_array[0]=(1000.0*(img_read_end-img_read_start))/CLOCKS_PER_SEC;
    
    std::clock_t img_present_check_start=std::clock();
    
    if (!image.data)
    {
        cout << "Image not found!\n";
        return -1;
    }
    
    std::clock_t img_present_check_end=std::clock();
    time_array[1]=(1000.0*(img_present_check_end-img_present_check_start))/CLOCKS_PER_SEC; 
    
    
    std::clock_t img_resize_check_start=std::clock();
    
    if(RESIZE_TO_DEBUG==1)
    {
      std::clock_t img_resize_start=std::clock();
      
      cv::resize(image,image,cv::Size(1024,1024));
      
      std::clock_t img_resize_end=std::clock();
      time_array[3]=(1000.0*(img_resize_end-img_resize_start))/CLOCKS_PER_SEC;
    }
    
    std::clock_t img_resize_check_end=std::clock();
    time_array[2]=(1000.0*(img_resize_check_end-img_resize_check_start))/CLOCKS_PER_SEC;

    // Read image dimensions
    std::clock_t img_dimensions_assign_start=std::clock();
    
    m = (uint16_t)image.rows; 
    n = (uint16_t)image.cols;
    uint16_t channels=(uint16_t)image.channels();  
    total=m*n;
    
    std::clock_t img_dimensions_assign_end=std::clock();
    time_array[4]=(1000.0*(img_dimensions_assign_end-img_dimensions_assign_start))/CLOCKS_PER_SEC;
    
    /*Declarations*/
    std::clock_t declarations_start=std::clock();
    
    uint8_t *random_array=(uint8_t*)malloc(sizeof(uint8_t)*256);
    uint8_t *img_xor_vec=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
    uint8_t *img_vec=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
    
    std::clock_t declarations_end=std::clock();
    time_array[5]=(1000.0*(declarations_end-declarations_start))/CLOCKS_PER_SEC;
    
    
    /*Print the image matrix*/
    std::clock_t check_print_images_start=std::clock();
    
    if(PRINT_IMAGES==1)
    {
      printImageContents(image); 
    }
   
    std::clock_t check_print_images_end=std::clock();
    time_array[6]=(1000.0*(check_print_images_end-check_print_images_start))/CLOCKS_PER_SEC;   

    /*Generate seed and system time value*/
    std::clock_t get_seed_start=std::clock();
    
    alpha=getSeed(1,32);
    
    std::clock_t get_seed_end=std::clock();
    time_array[7]=(1000.0*(get_seed_end-get_seed_start))/CLOCKS_PER_SEC;    

    std::clock_t sys_time_start=std::clock();
    
    manip_sys_time=(uint32_t)getManipulatedSystemTime();
    
    std::clock_t sys_time_end=std::clock();
    time_array[8]=(1000.0*(sys_time_end-sys_time_start))/CLOCKS_PER_SEC;
    
    
    /*Write seed and system time value to file*/
    std::clock_t write_parameters_start=clock();

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
    
    std::clock_t write_parameters_end=clock();
    time_array[9]=(1000.0*(write_parameters_end-write_parameters_start))/CLOCKS_PER_SEC;
 
    /*Generate PRNG*/
    std::clock_t generate_prng_start=std::clock();
    
    generatePRNG(random_array,alpha,manip_sys_time);
    
    std::clock_t generate_prng_end=std::clock();
    time_array[10]=(1000.0*(generate_prng_end-generate_prng_start))/CLOCKS_PER_SEC;

    /*Display generated PRNG*/
    if(DEBUG_VECTORS==1)
    {
      printf("\nrandom_array=");

      for(uint32_t i=0;i<256;++i)
      {
        printf("%d ",random_array[i]);
      }
    }    

    
    //Flattening Image, obtaining middle element and xor position
    std::clock_t flatten_image_start=std::clock();
    
    flattenImage(image,img_vec);
    
    std::clock_t flatten_image_end=std::clock();
    time_array[11]=(1000.0*(flatten_image_end-flatten_image_start))/CLOCKS_PER_SEC; 
    
    /*Display flattened image vector and write it to file*/
    if(DEBUG_VECTORS==1)
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
      
    
    }
    
    std::clock_t middle_element_assign_start=std::clock();
    
    middle_element=0;
     
    std::clock_t middle_element_assign_end=std::clock();
    time_array[12]=(1000.0*(middle_element_assign_end-middle_element_assign_start))/CLOCKS_PER_SEC;    

    std::clock_t position_assign_start=std::clock(); 
    
    xor_position=0;
    
    std::clock_t position_assign_end=std::clock();     
    time_array[13]=(1000.0*(position_assign_end-position_assign_start))/CLOCKS_PER_SEC;
    
    
    //printf("\nmiddle_element= %d",middle_element);
    //printf("\nxor_position= %d",xor_position);    

    //Xoring image vector
    std::clock_t xor_start=std::clock();

    xorImageDec(img_vec,img_xor_vec,m,n);
    
    /*Display XOR image vector*/
    if(DEBUG_VECTORS==1)
    {
      cout<<"\nDecrypted XOR vector=";
      for(int i=0;i<total*3;++i)
      {
        printf(" %d",img_xor_vec[i]);
      }
    }
    
    std::clock_t xor_end=std::clock();
    time_array[14]=(1000.0*(xor_end-xor_start))/CLOCKS_PER_SEC;    

    if(DEBUG_IMAGES==1)
    {
      std::clock_t img_vec_transfer_start=std::clock();
      
      /*for(int i=0;i<total*3;++i)
      {
        img_arr[i]=img_xor_vec[i];
      }*/
      
      std::clock_t img_vec_transfer_end=std::clock();
      time_array[15]=(1000.0*(img_vec_transfer_end-img_vec_transfer_start))/CLOCKS_PER_SEC;

      std::clock_t img_reshape_start=std::clock();
      
      cv::Mat img_reshape(m,n,CV_8UC3,img_xor_vec);

      std::clock_t img_reshape_end=std::clock();
      time_array[16]=(1000.0*(img_reshape_end-img_reshape_start))/CLOCKS_PER_SEC;  
      
      std::clock_t img_write_start=std::clock();
      
      cv::imwrite("airplane_decrypted.png",img_reshape);
      
      std::clock_t img_write_end=std::clock();
      time_array[17]=(1000.0*(img_write_end-img_write_start))/CLOCKS_PER_SEC;
    }
    
    /*Calculating total time and average time*/
    for(int i=0;i<18;++i)
    {
      total_time=total_time+time_array[i];
    }
    
    average_time=(total_time/18.00);
    
    cout<<"\nrows= "<<m;
    cout<<"\ncolumns="<<n;
    cout<<"\nchannels="<<channels;
    cout<<"\ntotal="<<total;
    printf("\nalpha= %d",alpha);
    printf("\nmanip_sys_time =%d",manip_sys_time);
    
    printf("\n\nRead image = %LF s",time_array[0]/1000.0);
    printf("\nCheck if image exists = %LF s",time_array[1]/1000.0);  
    printf("\nCheck if image is to be resized = %LF s",time_array[2]/1000.0);
    printf("\nResize image = %LF s",time_array[3]/1000.0);
    printf("\nAssign image dimensions = %LF s",time_array[4]/1000.0);
    printf("\nDeclare vectors and arrays = %LF s",time_array[5]/1000.0);    
    printf("\nCheck if image is to be printed = %LF s",time_array[6]/1000.0);
    printf("\nGet alpha = %LF s",time_array[7]/1000.0);
    printf("\nGet manipulated system time = %LF s",time_array[8]/1000.0);
    printf("\nWrite parameters to file = %LF s",time_array[9]/1000.0);
    printf("\nGenerate PRNG = %LF s",time_array[10]/1000.0);
    printf("\nFlatten image = %LF s",time_array[11]/1000.0);
    printf("\nGet middle element of PRNG = %LF s",time_array[12]/1000.0);
    printf("\nGet XOR starting position = %LF s",time_array[13]/1000.0);
    printf("\nSelf XOR image = %LF s",time_array[14]/1000.0);
    printf("\nTransfer image vector to image array = %LF s",time_array[15]/1000.0);
    printf("\nReshape image = %LF s",time_array[16]/1000.0);
    printf("\nWrite image = %LF s",time_array[17]/1000.0);                
    printf("\nTotal time = %LF s",total_time/1000.0);
    printf("\nAverage time = %LF s",average_time/1000.0);
                                
    return 0;
}


