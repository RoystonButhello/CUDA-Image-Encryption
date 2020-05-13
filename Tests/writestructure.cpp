#include "commonheader.hpp"
#include "pattern.hpp"
#include "io.hpp"
using namespace std;



int main()
{
  cout<<"\n\n\n\nIn Writestructure";
  FILE *outfile;
  uint32_t m = 2;
  uint32_t n = 2;
  uint32_t total = m * n;
  int count_map = 0;   
  uint8_t number_of_rounds = 5; 
  long ptr_position = 0;
  int fwrite_status = 9,fseek_status = 9;
  //(const uint8_t)common::getRandomUnsignedInteger8(ROUNDS_LOWER_LIMIT,ROUNDS_UPPER_LIMIT);
  
  /*Writing number of rounds to file*/
  cout<<"\n\nWRITING NUMBER OF ROUNDS TO FILE";
  
  outfile = fopen(config::constant_parameters_file_path,"wb");
  if(outfile == NULL)
   {
     cout<<"\nCould not open "<<config::parameters_file<<" from "<<config::rounds_file_path<<" for writing number of rounds";
     cout<<"\nExiting...";
     exit(0);
   }
  
  fwrite(&number_of_rounds,sizeof(uint8_t),1,outfile);
  //Getting pointer position after writing number of rounds
  ptr_position = ftell(outfile);
  cout<<"\npointer position after writing number of rounds = "<<ptr_position;
  fclose(outfile);
  
  int i = 0;
  config::algorithm rounds[number_of_rounds]; 
  config::lalm lalm_parameters[number_of_rounds];
  config::lasm lasm_parameters[number_of_rounds];
  config::slmm slmm_parameters[number_of_rounds];
  config::lma  lma_parameters[number_of_rounds];
  config::lm   lm_parameters[number_of_rounds];
  
  
  printf("\nSize of a round = %ld",sizeof(rounds[0]));
  printf("\nSize of an lm_parameter = %ld",sizeof(lm_parameters[0]));
  printf("\nSIze of an lma parameter = %ld",sizeof(lma_parameters[0]));
  printf("\nSize of an slmm_parameter = %ld",sizeof(slmm_parameters[0]));
  printf("\nSize of an lasm_parameter = %ld",sizeof(lasm_parameters[0]));
  printf("\nSize of an lalm_parameter = %ld",sizeof(lalm_parameters[0]));
  
  /*Initializing to zero*/
  cout<<"\n\nINITIALIZING ALL PARAMETERS TO ZERO";
  for( i = 0; i < number_of_rounds; ++i)
  {
    cout<<"\nRound "<<i;
    rounds[i].map = config::ChaoticMap(1);

    rounds[i].seed_lut_gen_1 = 0;
    rounds[i].seed_lut_gen_2 = 0;
    rounds[i].seed_row_rotate = 0;
    rounds[i].seed_col_rotate = 0;
    
    lalm_parameters[i].x_init = 0;
    lalm_parameters[i].y_init = 0;
    lalm_parameters[i].myu = 0;
    
    lasm_parameters[i].x_init = 0;
    lasm_parameters[i].y_init = 0;
    lasm_parameters[i].myu = 0;
  
    slmm_parameters[i].x_init = 0;
    slmm_parameters[i].y_init = 0;
    slmm_parameters[i].alpha = 0;
    slmm_parameters[i].beta = 0;
    
    lma_parameters[i].x_init = 0;
    lma_parameters[i].y_init = 0;
    lma_parameters[i].myu1 = 0;
    lma_parameters[i].myu2 = 0;
    lma_parameters[i].lambda1 = 0;
    lma_parameters[i].lambda2 = 0;
    
    lm_parameters[i].x_init = 0;
    lm_parameters[i].y_init = 0;
    lm_parameters[i].r = 0;
  }  
  
  
  printf("\nNumber of rounds = %d",number_of_rounds);
  /*Assigning basic parameters*/
  cout<<"\n\nASSIGNING BASIC PARAMETERS";
  for( i = 0; i < number_of_rounds; ++i)
  {
    cout<<"\nRound "<<i;
    rounds[i].map = common::mapAssigner(MAP_LOWER_LIMIT,MAP_UPPER_LIMIT);
    rounds[i].seed_lut_gen_1 = common::getRandomUnsignedInteger32(SEED_LOWER_LIMIT,SEED_UPPER_LIMIT);
    rounds[i].seed_lut_gen_2 = common::getRandomUnsignedInteger32(SEED_LOWER_LIMIT,SEED_UPPER_LIMIT);
    rounds[i].seed_row_rotate = common::getRandomUnsignedInteger32(SEED_LOWER_LIMIT,SEED_UPPER_LIMIT);
    rounds[i].seed_col_rotate = common::getRandomUnsignedInteger32(SEED_LOWER_LIMIT,SEED_UPPER_LIMIT);
    
    printf("\nmap = %d",int(rounds[i].map));
    printf("\nseed_lut_gen_1 = %d",rounds[i].seed_lut_gen_1);
    printf("\nseed_lut_gen_2 = %d",rounds[i].seed_lut_gen_2);
    printf("\nseed_row_rotate = %d",rounds[i].seed_row_rotate);
    printf("\nseed_col_rotate = %d",rounds[i].seed_col_rotate);    

   
    
  }  
   
   /*Writing basic parameters*/
   cout<<"\n\nWRITING BASIC PARAMETERS TO FILE";
   ptr_position = writeBasicParameters(outfile,config::constant_parameters_file_path,"ab",rounds,0,number_of_rounds,ptr_position);
   
  
  
  double *x = (double*)malloc(sizeof(double) * total * 3);
  double *y = (double*)malloc(sizeof(double) * total * 3);
  double *x_bar = (double*)malloc(sizeof(double) * total * 3);
  double *y_bar = (double*)malloc(sizeof(double) * total * 3);
  uint32_t *row_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *col_random_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *row_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *col_rotation_vec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *random_array = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  
  /*Assigning parameters to the selected map and running the selected map*/
  cout<<"\n\nASSIGNING PARAMETERS TO THE SELECTED MAP AND RUNNING IT";
  for(int i = 0; i < number_of_rounds; ++i)
  {
             cout<<"\nRound "<<i;
             if(int(rounds[i].map) == 1)
             {
               cout<<"\n\nGenerating random vectors through LM";
               lm_parameters[i].x_init = common::getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
               lm_parameters[i].y_init = common::getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
               lm_parameters[i].r = common::getRandomDouble(R_LOWER_LIMIT,R_UPPER_LIMIT);
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
              lma_parameters[i].x_init = common::getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
              lma_parameters[i].y_init = common::getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
              lma_parameters[i].myu1 = common::getRandomDouble(MYU1_LOWER_LIMIT,MYU1_UPPER_LIMIT);
              lma_parameters[i].myu2 = common::getRandomDouble(MYU2_LOWER_LIMIT,MYU2_UPPER_LIMIT);
              lma_parameters[i].lambda1 = common::getRandomDouble(LAMBDA1_LOWER_LIMIT,LAMBDA1_UPPER_LIMIT);
              lma_parameters[i].lambda2 = common::getRandomDouble(LAMBDA2_LOWER_LIMIT,LAMBDA2_UPPER_LIMIT);
              
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
              slmm_parameters[i].x_init = common::getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
              slmm_parameters[i].y_init = common::getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
              slmm_parameters[i].alpha = common::getRandomDouble(ALPHA_LOWER_LIMIT,ALPHA_UPPER_LIMIT);
              slmm_parameters[i].beta = common::getRandomDouble(BETA_LOWER_LIMIT,BETA_UPPER_LIMIT);
              
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
              lasm_parameters[i].x_init = common::getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
              lasm_parameters[i].y_init = common::getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
              lasm_parameters[i].myu = common::getRandomDouble(LASM_LOWER_LIMIT,LASM_UPPER_LIMIT);
              
              
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
              lalm_parameters[i].x_init = common::getRandomDouble(X_LOWER_LIMIT,X_UPPER_LIMIT);
              lalm_parameters[i].y_init = common::getRandomDouble(Y_LOWER_LIMIT,Y_UPPER_LIMIT);
              lalm_parameters[i].myu = common::getRandomDouble(MYU_LOWER_LIMIT,MYU_UPPER_LIMIT);
              
              
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
              cout<<"\nNo valid map selected for random vector generation\nExiting...";
              exit(0);
            }
             
 } 
  /*Writing map parameters to file*/
  cout<<"\n\nWRITING PARAMETERS TO FILE";
  for(int i = 0; i < number_of_rounds; ++i)
  {
    cout<<"\nRound "<<i;
    if(int(rounds[i].map) == 1)
    {
      cout<<"\nWriting lm parameters";
      ptr_position = writeLMParameters(outfile,config::constant_parameters_file_path,"ab",lm_parameters,i,number_of_rounds,ptr_position);
      
     }
     
    else if(int(rounds[i].map) == 2)
    {
      cout<<"\nWriting lma parameters";
      ptr_position = writeLMAParameters(outfile,config::constant_parameters_file_path,"ab",lma_parameters,i,number_of_rounds,ptr_position);
      
    }
    
    else if(int(rounds[i].map) == 3)
    {
      cout<<"\nWriting slmm parameters";
      ptr_position = writeSLMMParameters(outfile,config::constant_parameters_file_path,"ab",slmm_parameters,i,number_of_rounds,ptr_position);
      
    }
    
    else if(int(rounds[i].map) == 4)
    {
      cout<<"\nWriting lasm parameters";
      ptr_position = writeLASMParameters(outfile,config::constant_parameters_file_path,"ab",lasm_parameters,i,number_of_rounds,ptr_position);
    }
    
    else if(int(rounds[i].map) == 5)
    {
      cout<<"\nWriting lalm parameters";
      ptr_position = writeLALMParameters(outfile,config::constant_parameters_file_path,"ab",lalm_parameters,i,number_of_rounds,ptr_position);
    }
    
    else
    {
      cout<<"\nInvalid map option.\nExiting...";
      exit(0);
    }
   }
  return 0;
}
