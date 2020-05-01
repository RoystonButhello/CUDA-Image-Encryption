#include "include/commonheader.hpp"
#include "include/serial.hpp"
#include "include/pattern.hpp"
#include "include/kernel.hpp"
#include "include/io.hpp"

int main()
{
  cv::Mat image;
  
  
  image = cv::imread(config::diffused_image_path,cv::IMREAD_COLOR);
   
  if(!image.data)
  {
    cout<<"\nCould not open image from "<<"airplane_swapped_ROUND_3.png"<<" \nExiting...";
    exit(0);
  }

  if(RESIZE_TO_DEBUG == 1)
  {
    cv::resize(image,image,cv::Size(config::cols,config::rows),CV_INTER_LANCZOS4);
  }
  
  uint32_t m = 0,n = 0,channels = 0, total = 0;
  double alpha = 0.00,beta = 0.00;
  int number_of_rounds = 0;
  long ptr_position = 0; 
  m = image.rows;
  n = image.cols;
  channels = image.channels();
  total = m * n;
  int fread_status = 9,fseek_status = 9;  

  cout<<"\nRows = "<<m;
  cout<<"\nCols = "<<n;
  cout<<"\nChannels = "<<channels;
  
  
  /*Reading parameters*/
  FILE *infile = fopen(config::constant_parameters_file_path,"rb");
  
  if(infile == NULL)
  {
    cout<<"\nCould not open "<<config::constant_parameters_file_path<<" for reading\nExiting...";
    exit(0);
  }
  
  cout<<"\npointer position before reading the number of rounds = "<<number_of_rounds;

  fread_status = fread(&number_of_rounds,sizeof(number_of_rounds),1,infile);
  ptr_position =  ftell(infile);
  cout<<"\npointer position after reading the number of rounds = "<<ptr_position; 
  fclose(infile);
  
  cout<<"\nNumber of rounds = "<<number_of_rounds;
  
  /*Parameter arrays*/
  config::lm lm_parameters[number_of_rounds];
  config::lma lma_parameters[number_of_rounds];
  config::slmm slmm_parameters[number_of_rounds];
  config::lasm lasm_parameters[number_of_rounds];
  config::lalm lalm_parameters[number_of_rounds];
  config::mt mt_parameters[number_of_rounds];
  
  
  ptr_position = readMTParameters(infile,config::constant_parameters_file_path,"rb",mt_parameters,0,number_of_rounds,ptr_position);
  
  /*Display parameters after reading*/
  common::displayMapParameters(lm_parameters,lma_parameters,slmm_parameters,lasm_parameters,lalm_parameters,mt_parameters,number_of_rounds);
  
  uint8_t *enc_vec;
  uint8_t *dec_vec;
  uint8_t *final_vec;
  uint32_t *row_swap_lut_vec;
  uint32_t *col_swap_lut_vec;
  uint32_t *U;
  uint32_t *V;
  cudaMallocManaged((void**)&enc_vec,total * channels * sizeof(uint8_t));
  cudaMallocManaged((void**)&dec_vec,total * channels * sizeof(uint8_t));
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
  
  /*Vector generation for diffusion
  x[0] = 0.1;
  y[0] = 0.1;
  alpha = 1.00;
  beta = 3.00;
  pattern::twodSineLogisticModulationMap(x,y,diffusion_array,alpha,beta,total * channels);*/   

  /*Vector generation for row and column swapping
  common::genLUTVec(row_swap_lut_vec,m);
  common::genLUTVec(col_swap_lut_vec,n);
  pattern::MTSequence(row_random_vec,total * channels,config::lower_limit,config::upper_limit,config::seed_lut_gen_1);
  pattern::MTSequence(col_random_vec,total * channels,config::lower_limit,config::upper_limit,config::seed_lut_gen_2);
  common::rowColLUTGen(row_swap_lut_vec,row_random_vec,col_swap_lut_vec,col_random_vec,m,n);*/
  
  /*Flattening image*/
  common::flattenImage(image,enc_vec,channels);
  //cout<<"\nencrypted image = ";
  //common::printArray8(enc_vec,total * channels);
  
  if(DIFFUSION == 1)
  {
    cout<<"\nIn Undiffusion";
    pattern::MTSequence(diffusion_array,total * channels,config::lower_limit,config::upper_limit,mt_parameters[3].seed_5);
    serial::grayLevelTransform(enc_vec,diffusion_array,total * channels);

    if(DEBUG_VECTORS == 1)
    {
      printf("\n\nUndiffused image = ");
      common::printArray8(enc_vec,total * channels);
    }
     
    if(DEBUG_INTERMEDIATE_IMAGES == 1)
    {
      cv::Mat img_reshape(m,n,CV_8UC3,enc_vec);
      cv::imwrite(config::undiffused_image_path,img_reshape);
    }
  }  

  if(ROW_COL_SWAPPING == 1)
  {
    /*Row and Column Unswapping*/
    cout<<"\nIn row and column unswapping";
    for(int i = number_of_rounds - 1; i >= 0; --i)
    {
      
      pattern::MTSequence(row_random_vec,total * channels,config::lower_limit,config::upper_limit,mt_parameters[i].seed_3);
      pattern::MTSequence(col_random_vec,total * channels,config::lower_limit,config::upper_limit,mt_parameters[i].seed_4);
      
      common::genLUTVec(row_swap_lut_vec,m);
      common::genLUTVec(col_swap_lut_vec,n);
      common::rowColLUTGen(row_swap_lut_vec,row_random_vec,col_swap_lut_vec,col_random_vec,m,n);
      
      //cout<<"\n\ni = "<<i;
      //printf("\nBefore Swap");
      //printf("\nenc_vec = ");
      //common::printArray8(enc_vec,total * channels);
      //printf("\ndec_vec = ");
      //common::printArray8(dec_vec,total * channels);
      
      dim3 dec_row_col_swap_grid(m,n,1);
      dim3 dec_row_col_swap_blocks(channels,1,1);
      run_decRowColSwap(enc_vec,dec_vec,row_swap_lut_vec,col_swap_lut_vec,dec_row_col_swap_grid,dec_row_col_swap_blocks);
      
      //std::swap(enc_vec,dec_vec);
      /*for(int i = 0; i < total * channels; ++i) 
      {
        enc_vec[i] = dec_vec[i];
      }*/
      std::memcpy(enc_vec,dec_vec,total * channels);
          
      if(DEBUG_VECTORS == 1)
      {
        cout<<"\n\ni = "<<i;
        //printf("\nAfter Swap");
        printf("\nenc_vec = ");
        common::printArray8(enc_vec,total * channels);
        printf("\ndec_vec = ");
        common::printArray8(dec_vec,total * channels);
        cout<<"\nROUND = "<<i;
        /*cout<<"\n\ni = "<<i;
        printf("\nrow_swap_lut_vec = ");
        common::printArray32(row_swap_lut_vec,m);
        printf("\ncol_swap_lut_vec = ");
        common::printArray32(col_swap_lut_vec,n);*/
      }
  
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        config::unswapped_image = "";
        config::unswapped_image = config::image_name + "_unswapped" + "_ROUND_" + std::to_string(i) + ".png";
        cout<<"\n"<<config::unswapped_image;
        //printf("\nindex = %d",cnt);
        cv::Mat img_reshape(m,n,CV_8UC3,dec_vec);
        cv::imwrite(config::unswapped_image,img_reshape);  
      }
    }
  } 
  
  if(ROW_COL_ROTATION == 1)
  {
    /*Row and Column Unrotation*/
    cout<<"\nIn row and column unrotation ";
    for(int i = number_of_rounds - 1; i >=0; --i)
    {
      
      cout<<"\nROUND "<<i;
      

      /*Vector generation for row and column unrotation*/
      pattern::MTSequence(row_rotation_vec,total * channels,config::lower_limit,config::upper_limit,mt_parameters[i].seed_1);
      pattern::MTSequence(col_rotation_vec,total * channels,config::lower_limit,config::upper_limit,mt_parameters[i].seed_2);
      
      common::genLUTVec(U,m);
      common::genLUTVec(V,n);
      common::rowColLUTGen(U,row_rotation_vec,V,col_rotation_vec,m,n);
      
      
      
      //cout<<"\n\nBefore swap";
      //cout<<"\nenc_vec = ";
      //common::printArray8(enc_vec,total * channels);
      //printf("\ndec_vec = ");
      //common::printArray8(dec_vec,total * channels);
      dim3 dec_gen_cat_map_grid(m,n,1);
      dim3 dec_gen_cat_map_blocks(channels,1,1);
      run_DecGenCatMap(dec_vec,final_vec,V,U,dec_gen_cat_map_grid,dec_gen_cat_map_blocks);
      //std::swap(enc_vec,final_vec);
      //for(int i = 0; i < total * channels; ++i)
      //{
        //enc_vec[i] = final_vec[i];
      //}
      std::memcpy(dec_vec,final_vec,total * channels);
      if(DEBUG_VECTORS == 1)
      {
        cout<<"\ni = "<<i;
        //cout<<"\n\nAfter swap = ";
        printf("\ndec_vec = ");
        common::printArray8(dec_vec,total * channels);
        printf("\nfinal_vec = ");
        common::printArray8(final_vec,total * channels);
        /*cout<<"\nU = ";
        common::printArray32(U,m);
        cout<<"\nV = ";
        common::printArray32(V,n);*/
        /*cout<<"\nRow rotation vec = ";
        common::printArray32(row_rotation_vec,total * channels);
        cout<<"\nColumn rotation vec = ";
        common::printArray32(col_rotation_vec,total * channels);
        */
      }
  
      if(DEBUG_INTERMEDIATE_IMAGES == 1)
      {
        //printf("\nindex = %d",cnt);
        config::unrotated_image = "";
        config::unrotated_image = config::image_name + "_unrotated" + "_ROUND_" + std::to_string(i) + config::extension;
        cout<<"\n"<<config::unrotated_image;
        cv::Mat img_reshape(m,n,CV_8UC3,final_vec);
        int imwrite_status = cv::imwrite(config::unrotated_image,img_reshape);
        cout<<"\nimwrite_status = "<<imwrite_status;
      }
      //Swapping input and output vectors
      
      
    }  
  }
 
  return 0;
}
