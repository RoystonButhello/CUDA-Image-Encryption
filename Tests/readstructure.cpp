#include "commonheader.hpp"
#include "pattern.hpp"
#include "io.hpp"
using namespace std;



int main()
{ 
  cout<<"\n\n\n\nIn readstructure"; 
  FILE *infile;
  uint32_t m = 2;
  uint32_t n = 2;
  uint32_t total = m * n;     
  uint8_t number_of_rounds = 0;
  int count_map = 0;
  int i = 0;
  
  int fread_status = 9, fseek_status = 9;
  long ptr_position = 0;
  cout<<"\n\nREADING NUMBER OF ROUNDS FROM FILE";  
  
  infile = fopen(config::constant_parameters_file_path,"rb");
  if(infile == NULL)
  {
   cout<<"\nCould not open "<<config::parameters_file<<" from "<<config::parameters_file_path<<" for reading number of rounds";
   cout<<"\nExiting...";
   exit(0);
  }
  
  fread_status = fread(&number_of_rounds,sizeof(uint8_t),1,infile);
  ptr_position = ftell(infile);
  cout<<"\nfread status after reading the number of rounds = "<<fread_status;
  cout<<"\npointer position after reading the number of rounds = "<<ptr_position;
  fclose(infile);
  
  
  config::algorithm rounds[number_of_rounds]; 
  config::lalm lalm_parameters[number_of_rounds];
  config::lasm lasm_parameters[number_of_rounds];
  config::slmm slmm_parameters[number_of_rounds];
  config::lma  lma_parameters[number_of_rounds];
  config::lm   lm_parameters[number_of_rounds];
  
  printf("\nNumber of rounds = %d",number_of_rounds);
   

  /*Reading basic parameters*/
  cout<<"\n\nREADING BASIC PARAMETERS FROM FILE";
  ptr_position = readBasicParameters(infile,config::constant_parameters_file_path,"rb",rounds,0,number_of_rounds,ptr_position);
   
  cout<<"\n\nDISPLAYING BASIC PARAMETERS";
  for(int i = 0; i < number_of_rounds; ++i)
  {
    cout<<"\nRound "<<i;
    printf("\nmap = %d",int(rounds[i].map));
    printf("\nseed_lut_gen_1 = %d",rounds[i].seed_lut_gen_1);
    printf("\nseed_lut_gen_2 = %d",rounds[i].seed_lut_gen_2);
    printf("\nseed_row_rotate = %d",rounds[i].seed_row_rotate);
    printf("\nseed_col_rotate = %d",rounds[i].seed_col_rotate);
  } 

  
  double *x = (double*)malloc(sizeof(double) * total * 3);
  double *y = (double*)malloc(sizeof(double) * total * 3);
  double *x_bar = (double*)malloc(sizeof(double) * total * 3);
  double *y_bar = (double*)malloc(sizeof(double) * total * 3);
  uint32_t *row_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *col_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *row_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *col_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *random_array = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  
  /*Reading chaotic map parameters*/
  cout<<"\n\nREADING CHAOTIC MAP PARAMETERS FROM FILE";
  for(int i = 0; i < number_of_rounds; ++i)
  {
    cout<<"\nRound "<<i;
    if(int(rounds[i].map) == 1)
    {
      cout<<"\nReading lm parameters";
      ptr_position = readLMParameters(infile,config::constant_parameters_file_path,"rb",lm_parameters,i,number_of_rounds,ptr_position);
 
    }
  
    else if(int(rounds[i].map) == 2)
    {
      cout<<"\nReading lma parameters";
      ptr_position = readLMAParameters(infile,config::constant_parameters_file_path,"rb",lma_parameters,i,number_of_rounds,ptr_position);
      
    }
  
    else if(int(rounds[i].map) == 3)
    {
      cout<<"\nReading slmm parameters";
      ptr_position = readSLMMParameters(infile,config::constant_parameters_file_path,"rb",slmm_parameters,i,number_of_rounds,ptr_position);
    }
    
    else if(int(rounds[i].map) == 4)
    {
      cout<<"\nReading lasm parameters";
      ptr_position = readLASMParameters(infile,config::constant_parameters_file_path,"rb",lasm_parameters,i,number_of_rounds,ptr_position);
    }
    
    else if(int(rounds[i].map) == 5)
    {
      cout<<"\nReading lalm parameters";
      ptr_position = readLALMParameters(infile,config::constant_parameters_file_path,"rb",lalm_parameters,i,number_of_rounds,ptr_position);
    }
    
    else
    {
      cout<<"\nInvalid map read option\nExiting...";
      exit(0);
    }
   } 
    
  
  /*Running the selected map*/
  cout<<"\nRUNNING THE SELECTED CHAOTIC MAP";
  for(int i = 0; i < number_of_rounds; ++i)
  {
    cout<<"\nRound "<<i;
    if(int(rounds[i].map) == 1)
    { 
               
      cout<<"\n\nGenerating random vectors through LM";
      x[0] = lm_parameters[i].x_init;
      y[0] = lm_parameters[i].y_init;
               
      pattern::twodLogisticMap(x,y,row_random_vec,lm_parameters[i].r,total * 3);
      pattern::twodLogisticMap(x,y,col_random_vec,lm_parameters[i].r,total * 3);
      pattern::twodLogisticMap(x,y,row_rotation_vec,lm_parameters[i].r,total * 3);
      pattern::twodLogisticMap(x,y,col_rotation_vec,lm_parameters[i].r,total * 3);
             
      if(DEBUG_FILE_CONTENTS == 1)
      {
        printf("\ni = %d",i);
        printf("\nx_init = %f",lm_parameters[i].x_init);
        printf("\ny_init = %f",lm_parameters[i].y_init);
        printf("\nr = %f",lm_parameters[i].r);    
      }
       
      if(DEBUG_VECTORS == 1)
      {
          printf("\nx = ");
          common::printArrayDouble(x,total * 3);
          printf("\ny = ");
          common::printArrayDouble(y,total * 3);
          printf("\nrow_random_vec =");
          common::printArray32(row_random_vec,total * 3);
          printf("\ncol_random_vec = ");
          common::printArray32(col_random_vec,total * 3);
          printf("\nrow_rotation_vec = ");
          common::printArray32(row_rotation_vec,total * 3);
          printf("\ncol_rotation_vec = ");
          common::printArray32(col_rotation_vec,total * 3);  
      }
                
    }
    
    else if(int(rounds[i].map) == 2)
    {
       cout<<"\n\nGenerating random vectors through LMA";
       x[0] = lma_parameters[i].x_init;
       y[0] = lma_parameters[i].y_init;
              
       pattern::twodLogisticMapAdvanced(x,y,row_random_vec,lma_parameters[i].myu1,lma_parameters[i].myu2,lma_parameters[i].lambda1,lma_parameters[i].lambda2,total * 3);
       pattern::twodLogisticMapAdvanced(x,y,col_random_vec,lma_parameters[i].myu1,lma_parameters[i].myu2,lma_parameters[i].lambda1,lma_parameters[i].lambda2,total * 3);
       pattern::twodLogisticMapAdvanced(x,y,row_rotation_vec,lma_parameters[i].myu1,lma_parameters[i].myu2,lma_parameters[i].lambda1,lma_parameters[i].lambda2,total * 3);
       pattern::twodLogisticMapAdvanced(x,y,col_rotation_vec,lma_parameters[i].myu1,lma_parameters[i].myu2,lma_parameters[i].lambda1,lma_parameters[i].lambda2,total * 3);
              
       if(DEBUG_FILE_CONTENTS == 1)
       { 
         printf("\ni = %d",i);
         printf("\nx_init = %f",lma_parameters[i].x_init);
         printf("\ny_init = %f",lma_parameters[i].y_init);
         printf("\nmyu1 = %f",lma_parameters[i].myu1);
         printf("\nmyu2 = %f",lma_parameters[i].myu2);
         printf("\nlambda1 = %f",lma_parameters[i].lambda1);
         printf("\nlambda2 = %f",lma_parameters[i].lambda2);
       }
      
      if(DEBUG_VECTORS == 1)
      {
          printf("\nx = ");
          common::printArrayDouble(x,total * 3);
          printf("\ny = ");
          common::printArrayDouble(y,total * 3);
          printf("\nrow_random_vec =");
          common::printArray32(row_random_vec,total * 3);
          printf("\ncol_random_vec = ");
          common::printArray32(col_random_vec,total * 3);
          printf("\nrow_rotation_vec = ");
          common::printArray32(row_rotation_vec,total * 3);
          printf("\ncol_rotation_vec = ");
          common::printArray32(col_rotation_vec,total * 3);  
      }
    }
          
    else if(int(rounds[i].map) == 3)
    {
      cout<<"\n\nGenerating random vectors through SLMM";
              
      x[0] = slmm_parameters[i].x_init;
      y[0] = slmm_parameters[i].y_init;
             
      pattern::twodSineLogisticModulationMap(x,y,row_random_vec,slmm_parameters[i].alpha,slmm_parameters[i].beta,total * 3);
      pattern::twodSineLogisticModulationMap(x,y,col_random_vec,slmm_parameters[i].alpha,slmm_parameters[i].beta,total * 3);
      pattern::twodSineLogisticModulationMap(x,y,row_rotation_vec,slmm_parameters[i].alpha,slmm_parameters[i].beta,total * 3);
      pattern::twodSineLogisticModulationMap(x,y,col_rotation_vec,slmm_parameters[i].alpha,slmm_parameters[i].beta,total * 3);

      if(DEBUG_FILE_CONTENTS == 1)
      { 
        printf("\ni = %d",i);
        printf("\nx_init = %f",slmm_parameters[i].x_init);
        printf("\ny_init = %f",slmm_parameters[i].y_init);
        printf("\nalpha = %f",slmm_parameters[i].alpha);
        printf("\nbeta = %f",slmm_parameters[i].beta);
      }
      
      if(DEBUG_VECTORS == 1)
      {
          printf("\nx = ");
          common::printArrayDouble(x,total * 3);
          printf("\ny = ");
          common::printArrayDouble(y,total * 3);
          printf("\nrow_random_vec =");
          common::printArray32(row_random_vec,total * 3);
          printf("\ncol_random_vec = ");
          common::printArray32(col_random_vec,total * 3);
          printf("\nrow_rotation_vec = ");
          common::printArray32(row_rotation_vec,total * 3);
          printf("\ncol_rotation_vec = ");
          common::printArray32(col_rotation_vec,total * 3);  
      }        
    }
     
    else if(int(rounds[i].map) == 4)
    {
       cout<<"\n\nGenerating random vectors through LASM";
         
       x[0] = lasm_parameters[i].x_init;
       y[0] = lasm_parameters[i].y_init;
             
       pattern::twodLogisticAdjustedSineMap(x,y,row_random_vec,lasm_parameters[i].myu,total * 3);
       pattern::twodLogisticAdjustedSineMap(x,y,col_random_vec,lasm_parameters[i].myu,total * 3);
       pattern::twodLogisticAdjustedSineMap(x,y,row_rotation_vec,lasm_parameters[i].myu,total * 3);
       pattern::twodLogisticAdjustedSineMap(x,y,col_rotation_vec,lasm_parameters[i].myu,total * 3);

       if(DEBUG_FILE_CONTENTS == 1)
       { 
         printf("\ni = %d",i);
         printf("\nx_init = %f",lasm_parameters[i].x_init);
         printf("\ny_init = %f",lasm_parameters[i].y_init);
         printf("\nmyu = %f",lasm_parameters[i].myu);
       }
             
       if(DEBUG_VECTORS == 1)
       {
         printf("\nx = ");
         common::printArrayDouble(x,total * 3);
         printf("\ny = ");
         common::printArrayDouble(y,total * 3);
    
         printf("\nrow_random_vec =");
         common::printArray32(row_random_vec,total * 3);
         printf("\ncol_random_vec = ");
         common::printArray32(col_random_vec,total * 3);
         printf("\nrow_rotation_vec = ");
         common::printArray32(row_rotation_vec,total * 3);
         printf("\ncol_rotation_vec = ");
         common::printArray32(col_rotation_vec,total * 3);  
       }
     }
     
    else if(int(rounds[i].map) == 5)
    {
       cout<<"\n\nGenerating random vectors through LALM";
         
       x[0] = lalm_parameters[i].x_init;
       y[0] = lalm_parameters[i].y_init;
             
       pattern::twodLogisticAdjustedLogisticMap(x,y,x_bar,y_bar,row_random_vec,lalm_parameters[i].myu,total * 3);
       pattern::twodLogisticAdjustedLogisticMap(x,y,x_bar,y_bar,col_random_vec,lalm_parameters[i].myu,total * 3);
       pattern::twodLogisticAdjustedLogisticMap(x,y,x_bar,y_bar,row_rotation_vec,lalm_parameters[i].myu,total * 3);
       pattern::twodLogisticAdjustedLogisticMap(x,y,x_bar,y_bar,col_rotation_vec,lalm_parameters[i].myu,total * 3);

       if(DEBUG_FILE_CONTENTS == 1)
       { 
         printf("\ni = %d",i);
         printf("\nx_init = %f",lalm_parameters[i].x_init);
         printf("\ny_init = %f",lalm_parameters[i].y_init);
         printf("\nmyu = %f",lalm_parameters[i].myu);
       }
             
       if(DEBUG_VECTORS == 1)
       {
         printf("\nx = ");
         common::printArrayDouble(x,total * 3);
         printf("\ny = ");
         common::printArrayDouble(y,total * 3);
    
         printf("\nrow_random_vec =");
         common::printArray32(row_random_vec,total * 3);
         printf("\ncol_random_vec = ");
         common::printArray32(col_random_vec,total * 3);
         printf("\nrow_rotation_vec = ");
         common::printArray32(row_rotation_vec,total * 3);
         printf("\ncol_rotation_vec = ");
         common::printArray32(col_rotation_vec,total * 3);  
       }
     } 
     
   else
   {
     cout<<"\nInvalid map option";
   }          
      
  }
  
  return 0;
}
