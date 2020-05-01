#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"
#include "include/kernel.hpp"
#include "include/io.hpp"


int main()
{
  cv::Mat image;
  image = cv::imread(config::input_image_path,cv::IMREAD_COLOR);
  
  if(!image.data)
  {
    cout<<"\nCould not open image from "<<config::input_image_path<<" \nExiting...";
    exit(0);
  }

  if(RESIZE_TO_DEBUG == 1)
  {
    cv::resize(image,image,cv::Size(config::cols,config::rows),CV_INTER_LANCZOS4);
  }
  
  
  uint32_t m = 0,n = 0,channels = 0, total = 0;
  int number_of_rounds = 4;
  int fseek_status = 9,fwrite_status = 9;
  
  long ptr_position = 0;
  double alpha = 0.00,beta = 0.00;
  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;
  //cv::Mat4b img_dec(m,n);
  cout<<"\nRows = "<<m;
  cout<<"\nCols = "<<n;
  cout<<"\nChannels = "<<channels;
  
  
  /*Parameter arrays*/
  config::lm lm_parameters[number_of_rounds];
  config::lma lma_parameters[number_of_rounds];
  config::slmm slmm_parameters[number_of_rounds];
  config::lasm lasm_parameters[number_of_rounds];
  config::lalm lalm_parameters[number_of_rounds];
  config::mt mt_parameters[number_of_rounds];
  
  /*Initializing parameters to zero for each round*/
  common::initializeMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,number_of_rounds);
  
  /*Assigning parameters for each round*/
  common::assignMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,number_of_rounds);
  
  /*Writing parameters*/
  FILE *outfile = fopen(config::constant_parameters_file_path,"wb");
  
  if(outfile == NULL)
  {
    cout<<"\nCould not open "<<config::constant_parameters_file_path<<" for writing\nExiting...";
    exit(0);
  }
  
  cout<<"\npointer position before writing the number of rounds = "<<number_of_rounds;

  fwrite_status = fwrite(&number_of_rounds,sizeof(number_of_rounds),1,outfile);
  ptr_position =  ftell(outfile);
  cout<<"\nfwrite status after writing the number of rounds = "<<fwrite_status;
  cout<<"\npointer position after writing the number of rounds = "<<ptr_position; 
  fclose(outfile);
  
  cout<<"\nNumber of rounds = "<<number_of_rounds;
  
  
  ptr_position = writeMTParameters(outfile,config::constant_parameters_file_path,"ab",mt_parameters,0,number_of_rounds,ptr_position);
  
  /*Display parameters after writing*/
  common::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,number_of_rounds);
  
  uint8_t *img_vec;
  uint8_t *enc_vec;
  uint8_t *final_vec;
  uint32_t *row_swap_lut_vec;
  uint32_t *col_swap_lut_vec;
  uint32_t *U;
  uint32_t *V;
  cudaMallocManaged((void**)&img_vec,total * channels * sizeof(uint8_t));
  cudaMallocManaged((void**)&enc_vec,total * channels * sizeof(uint8_t));
  cudaMallocManaged((void**)&final_vec,total * channels * sizeof(uint8_t));
  cudaMallocManaged((void**)&row_swap_lut_vec,m * sizeof(uint32_t));
  cudaMallocManaged((void**)&col_swap_lut_vec,n * sizeof(uint32_t));
  cudaMallocManaged((void**)&U,m * sizeof(uint32_t));
  cudaMallocManaged((void**)&V,n * sizeof(uint32_t));
  
  double *x = (double*)malloc(sizeof(double) * total * channels);
  double *y = (double*)malloc(sizeof(double) * total * channels);
  uint32_t *row_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *col_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *row_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *col_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  uint32_t *diffusion_array = (uint32_t*)malloc(sizeof(uint32_t) * total * channels);
  
 

  /*Vector generation for row and column swapping
  common::genLUTVec(row_swap_lut_vec,m);
  common::genLUTVec(col_swap_lut_vec,n);
  pattern::MTSequence(row_random_vec,total * channels,config::lower_limit,config::upper_limit,config::seed_lut_gen_1);
  pattern::MTSequence(col_random_vec,total * channels,config::lower_limit,config::upper_limit,config::seed_lut_gen_2);
  common::rowColLUTGen(row_swap_lut_vec,row_random_vec,col_swap_lut_vec,col_random_vec,m,n);*/
  
  /*Flattening image*/
  common::flattenImage(image,img_vec,channels);
  //cout<<"\nplain image = ";
  //common::printArray8(img_vec,total * channels);
  
  if(ROW_COL_ROTATION == 1)
  {
    cout<<"\nIn row and column rotation";
    for(int i = 0; i < number_of_rounds; ++i)
    {
      cout<<"\nROUND "<<i;
      
      /*Vector generation for row and column rotation*/
      pattern::MTSequence(row_rotation_vec,total * channels,config::lower_limit,config::upper_limit,mt_parameters[i].seed_1);
      pattern::MTSequence(col_rotation_vec,total * channels,config::lower_limit,config::upper_limit,mt_parameters[i].seed_2);
      
      common::genLUTVec(U,m);
      common::genLUTVec(V,n);
      common::rowColLUTGen(U,row_rotation_vec,V,col_rotation_vec,m,n);  
      
      /*Row and Column Rotation*/
      
      //cout<<"\n\nBefore swap";
      //cout<<"\nimg_vec = ";
      //common::printArray8(img_vec,total * channels);
      //printf("\nenc_vec = ");
      //common::printArray8(enc_vec,total * channels);
      dim3 enc_gen_cat_map_grid(m,n,1);
      dim3 enc_gen_cat_map_blocks(channels,1,1);
      run_EncGenCatMap(img_vec,enc_vec,V,U,enc_gen_cat_map_grid,enc_gen_cat_map_blocks);
      
      //for(int i = 0; i < total * channels; ++i)
      //{
        //img_vec[i] = enc_vec[i];
      //}
      
      std::memcpy(img_vec,enc_vec,total * channels);
      //std::swap(img_vec,enc_vec);
      
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        config::rotated_image = "";
        config::rotated_image = config::image_name + "_rotated" + "_ROUND_" + std::to_string(i) + config::extension;
        
        cout<<"\n"<<config::rotated_image;
        cv::Mat img_reshape(m,n,CV_8UC3,enc_vec);
        cv::imwrite(config::rotated_image,img_reshape);
        
      }
      
      
      if(DEBUG_VECTORS == 1)
      {
        //cout<<"\n\nAfter swap";
        cout<<"\nimg_vec = ";
        common::printArray8(img_vec,total * channels);
        printf("\nenc_vec = ");
        common::printArray8(enc_vec,total * channels);
        /*cout<<"\nU = ";
        common::printArray32(U,m);
        cout<<"\nV = ";
        common::printArray32(V,n);*/
        /*cout<<"\nRow rotation vec = ";
        common::printArray32(row_rotation_vec,total * channels);
        cout<<"\nColumn rotation vec = ";
        common::printArray32(col_rotation_vec,total * channels);*/
      }
      //Swapping input and output vectors
      
    }
 } 
  
  if(ROW_COL_SWAPPING == 1)
  {
    /*Row and Column Swapping*/
    cout<<"\nIn row and column swapping";
    for(int i = 0; i < number_of_rounds; ++i)
    {
      /*Vector generation for row and coulumn swapping*/
      pattern::MTSequence(row_random_vec,total * channels,config::lower_limit,config::upper_limit,mt_parameters[i].seed_3);
      pattern::MTSequence(col_random_vec,total * channels,config::lower_limit,config::upper_limit,mt_parameters[i].seed_4);
      
      common::genLUTVec(row_swap_lut_vec,m);
      common::genLUTVec(col_swap_lut_vec,n);
      common::rowColLUTGen(row_swap_lut_vec,row_random_vec,col_swap_lut_vec,col_random_vec,m,n);
      
      //cout<<"\n\ni = "<<i;
      //printf("\nBefore Swap");
      //printf("\nimg_vec = ");
      //common::printArray8(img_vec,total * channels);
      //printf("\nfinal_vec = ");
      //common::printArray8(final_vec,total * channels);
      
      dim3 enc_row_col_swap_grid(m,n,1);
      dim3 enc_row_col_swap_blocks(channels,1,1);
      run_encRowColSwap(enc_vec,final_vec,row_swap_lut_vec,col_swap_lut_vec,enc_row_col_swap_grid,enc_row_col_swap_blocks);
      
      /*if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        //if(i%4 == 0)
        //{
          config::swapped_image = "";
          config::swapped_image = config::image_name + "_swapped" + "_ROUND_" + std::to_string(i) + config::extension;
          cout<<"\n"<<config::swapped_image;  
          cv::Mat img_reshape(m,n,CV_8UC3,final_vec);
          cv::imwrite(config::swapped_image,img_reshape);
        //}
      }*/
      
      //std::swap(img_vec,final_vec);
      
      /*for(int i = 0; i < total * channels; ++i)
      {
        img_vec[i] = final_vec[i];
      }*/
      
      std::memcpy(enc_vec,final_vec,total * channels);
      
      if(DEBUG_VECTORS == 1)
      {
        cout<<"\n\ni = "<<i;
        //printf("\nAfter Swap");
        printf("\nenc_vec = ");
        common::printArray8(enc_vec,total * channels);
        printf("\nfinal_vec = ");
        common::printArray8(final_vec,total * channels);
        /*cout<<"\n\ni = "<<i;
        printf("\nrow_swap_lut_vec = ");
        common::printArray32(row_swap_lut_vec,m);
        printf("\ncol_swap_lut_vec = ");
        common::printArray32(col_swap_lut_vec,n);*/
      }
  
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        //if(i%4 != 0)
        //{
          config::swapped_image = "";
          config::swapped_image = config::image_name + "_swapped" + "_ROUND_" + std::to_string(i) + config::extension;
          cout<<"\n"<<config::swapped_image;  
          cv::Mat img_reshape(m,n,CV_8UC3,final_vec);
          cv::imwrite(config::swapped_image,img_reshape);
        //}    
      }
    }
  }
  
  if(DIFFUSION == 1)
  {
    
    cout<<"\nIn Diffusion";
    
    pattern::MTSequence(diffusion_array,total * channels,config::lower_limit,config::upper_limit,mt_parameters[3].seed_5);
    serial::grayLevelTransform(final_vec,diffusion_array,total * channels);

    if(DEBUG_VECTORS == 1)
    {
      printf("\n\nRotated Swapped and Diffused image = ");
      common::printArray8(final_vec,total * channels);
    }
     
    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      cv::Mat img_reshape(m,n,CV_8UC3,final_vec);
      cv::imwrite(config::diffused_image_path,img_reshape);
    }
  }
  return 0;
}
