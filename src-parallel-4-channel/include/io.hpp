#ifndef IO_H
#define IO_H


  static inline long rwLMParameters(FILE *outfile,const char *file_path,const char *mode,config::lm lm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long rwLMAParameters(FILE *outfile,const char *file_path,const char *mode,config::lma lma_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long rwSLMMParameters(FILE *outfile,const char *file_path,const char *mode,config::slmm slmm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long rwLASMParameters(FILE *outfile,const char *file_path,const char *mode,config::lasm lasm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long rwLALMParameters(FILE *outfile,const char *file_path,const char *mode,config::lalm lalm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long rwMTParameters(FILE *outfile,const char *file_path,const char *mode,config::mt mt_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long rwIntegerArray(FILE *outfile,const char *file_path,const char *mode,uint32_t *map_choice_array,int length,long ptr_position);
  void rwInteger(FILE *outfile,const char *file_path,const char *mode,int seed,int length,long ptr_position);
  

static inline long rwLMParameters(FILE *outfile,const char *file_path,const char *mode,config::lm lm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    long pointer_position = ptr_position;
    int fwrite_status = 9,fseek_status = 9,fread_status = 9;
    outfile = fopen(file_path,mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    
    //Offset pointer position by length of previous record
    if(strcmp("ab",mode) == 0 || strcmp("wb",mode) == 0)
    {
      if(pointer_position > 0)
      {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
      }
      
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before writing lm_parameters = "<<fseek_status;
        cout<<"\npointer position before writing lm_parameters = "<<pointer_position;
      }
      
      //Write lm parameters to file
      size_t size = number_of_rounds * sizeof(lm_parameters[0]);
      fwrite_status = fwrite(lm_parameters,size,1,outfile);
      //Update pointer position after writing
      pointer_position = ftell(outfile);
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfwrite status after writing lm_parameters = "<<fwrite_status;
        cout<<"\npointer position after writing lm_parameters = "<<pointer_position;
      }
      
    }
    
    else
    {
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\npointer position before reading lm_parameters = "<<pointer_position;
      }
      
      //Ofsetting pointer position by length of revious record
      if(pointer_position > 0)
      {
        fseek_status = fseek(outfile,(pointer_position),SEEK_SET);
        pointer_position = ftell(outfile);
      }
      
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before reading lm_parameters = "<<fseek_status;
        cout<<"\npointer position before writing lm_parameters = "<<pointer_position;
      }
      
      //Read lm parameters from file
      size_t size = number_of_rounds * sizeof(lm_parameters[0]);
      fread_status = fread(lm_parameters,size,1,outfile);
      //Update pointer position after reading
      pointer_position = ftell(outfile);
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfread status after reading lm_parameters = "<<fread_status;
        cout<<"\npointer position after reading lm_parameters = "<<pointer_position;
      }
    }
    
    fclose(outfile);
    return pointer_position;
  }
  
  static inline long rwLMAParameters(FILE *outfile,const char *file_path,const char *mode,config::lma lma_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    long pointer_position = ptr_position;
    int fwrite_status = 9,fseek_status = 9,fread_status = 9;
    outfile = fopen(file_path,mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    if(strcmp("wb",mode) == 0 || strcmp("ab",mode) == 0)
    {
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before writing lma_parameters = "<<fseek_status;
        cout<<"\npointer position before writing lma_parameters = "<<pointer_position;
      }
    
      //Write lma parameters to file
      size_t size = sizeof(lma_parameters[0]) * number_of_rounds;
      fwrite_status = fwrite(lma_parameters,size,1,outfile);
      //Update pointer position after writing
      pointer_position = ftell(outfile);
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfwrite status after writing lma_parameters = "<<fwrite_status;
        cout<<"\npointer position after writing lma_parameters = "<<pointer_position;
      }
    }
    
    else
    {
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\npointer position before reading lma_parameters = "<<pointer_position;
      }
      
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
        fseek_status = fseek(outfile,(pointer_position),SEEK_SET);
        pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before reading lma_parameters = "<<fseek_status;
        cout<<"\npointer position before reading lma_parameters = "<<pointer_position;
      }
    
      //Read lma parameters from file
      size_t size = sizeof(lma_parameters[0]) * number_of_rounds;
      fread_status = fread(lma_parameters,size,1,outfile);
      //Update pointer position after reading
      pointer_position = ftell(outfile);
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfread status after writing lma_parameters = "<<fread_status;
        cout<<"\npointer position after reading lma_parameters = "<<pointer_position;
      }
    }
    
    fclose(outfile);
    return pointer_position;
  }
  
  
  static inline long rwSLMMParameters(FILE *outfile,const char *file_path,const char *mode,config::slmm slmm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    long pointer_position = ptr_position;
    int fwrite_status = 9,fseek_status = 9,fread_status = 9;
    outfile = fopen(file_path,mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    if(strcmp("wb",mode) == 0 || strcmp("ab",mode) == 0)
    {
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
         fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
         pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before writing slmm_parameters = "<<fseek_status;
        cout<<"\npointer position before writing slmm_parameters = "<<pointer_position;
      }
      
      //Write slmm parameters to file
      size_t size = sizeof(slmm_parameters[0]) * number_of_rounds;
      fwrite_status = fwrite(slmm_parameters,size,1,outfile);
      //Update pointer position after writing
      pointer_position = ftell(outfile);
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfwrite status after writing slmm_parameters = "<<fwrite_status;
        cout<<"\npointer position after writing slmm_parameters = "<<pointer_position;
      }
    }
    
    else
    {
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\npointer position before reading slmm_parameters = "<<pointer_position;
      }
      
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
         fseek_status = fseek(outfile,(pointer_position),SEEK_SET);
         pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before reading slmm_parameters = "<<fseek_status;
        cout<<"\npointer position before reading slmm_parameters = "<<pointer_position;
      }
      
      //Read slmm parameters from file
      size_t size = sizeof(slmm_parameters[0]) * number_of_rounds;
      fread_status = fread(slmm_parameters,size,1,outfile);
      //Update pointer position after reading
      pointer_position = ftell(outfile);
      
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfread status after reading slmm_parameters = "<<fread_status;
        cout<<"\npointer position after reading slmm_parameters = "<<pointer_position;
      }  
    } 
    
    fclose(outfile);
    return pointer_position;
  }
  
  static inline long rwLASMParameters(FILE *outfile,const char *file_path,const char *mode,config::lasm lasm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    long pointer_position = ptr_position;
    int fwrite_status = 9,fseek_status = 9,fread_status = 9;
    outfile = fopen(file_path,mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    if(strcmp("wb",mode) == 0 || strcmp("ab",mode) == 0)
    {
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
         fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
         pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before writing lasm_parameters = "<<fseek_status;
        cout<<"\npointer position before writing lasm_parameters = "<<pointer_position;
      }
    
      //Write lasm parameters to file
      size_t size = sizeof(lasm_parameters[0]) * number_of_rounds;
      fwrite_status = fwrite(lasm_parameters,size,1,outfile);
      //Update pointer position after writing
      pointer_position = ftell(outfile);
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfwrite status after writing lasm_parameters = "<<fwrite_status;
        cout<<"\npointer position after writing lasm_parameters = "<<pointer_position;
      }
    }
    
    else
    {
      
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\npointer position before writing lasm_parameters = "<<pointer_position;
      }
      
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
         fseek_status = fseek(outfile,(pointer_position),SEEK_SET);
         pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before reading lasm_parameters = "<<fseek_status;
        cout<<"\npointer position before reading lasm_parameters = "<<pointer_position;
      }
    
      //Read lasm parameters from file
      size_t size = sizeof(lasm_parameters[0]) * number_of_rounds;
      fread_status = fread(lasm_parameters,size,1,outfile);
      //Update pointer position after reading
      pointer_position = ftell(outfile);
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfread status after reading lasm_parameters = "<<fread_status;
        cout<<"\npointer position after reading lasm_parameters = "<<pointer_position;
      }
    }
    
    fclose(outfile);
    return pointer_position;
  } 
  
  static inline long rwLALMParameters(FILE *outfile,const char *file_path,const char *mode,config::lalm lalm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    long pointer_position = ptr_position;
    int fwrite_status = 9,fseek_status = 9,fread_status = 9;
    outfile = fopen(file_path,mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    if(strcmp("wb",mode) == 0 || strcmp("ab",mode) == 0)
    {
      
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
         fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
         pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before writing lalm_parameters = "<<fseek_status;
        cout<<"\npointer position before writing lalm_parameters = "<<pointer_position;
      }
    
      //Write lalm parameters to file
      size_t size = sizeof(lalm_parameters[0]) * number_of_rounds;
      fwrite_status = fwrite(lalm_parameters,size,1,outfile);
      //Update pointer position after writing
      pointer_position = ftell(outfile);
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfwrite status after writing lalm_parameters = "<<fwrite_status;
        cout<<"\npointer position after writing lalm_parameters = "<<pointer_position;
      }
    }
    
    else
    {
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\npointer position before reading lalm_parameters  = "<<pointer_position;
      }
      
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
         fseek_status = fseek(outfile,(pointer_position),SEEK_SET);
         pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before reading lalm_parameters = "<<fseek_status;
        cout<<"\npointer position before reading lalm_parameters = "<<pointer_position;
      }
    
      //Read lalm parameters from file
      size_t size = sizeof(lalm_parameters[0]) * number_of_rounds;
      fread_status = fread(lalm_parameters,size,1,outfile);
      //Update pointer position after reading
      pointer_position = ftell(outfile);
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfread status after writing lalm_parameters = "<<fread_status;
        cout<<"\npointer position after reading lalm_parameters = "<<pointer_position;
      }  
    }
    
    fclose(outfile);
    return pointer_position;
  }
  
  static inline long rwMTParameters(FILE *outfile,const char *file_path,const char *mode,config::mt mt_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    int fseek_status = 9,fwrite_status = 9,fread_status = 9;
    long pointer_position = ptr_position;
    outfile = fopen(file_path,mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    if(strcmp("wb",mode) == 0 || strcmp("ab",mode) == 0)
    {
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before writing basic parameters = "<<fseek_status;
        cout<<"\npointer position before writing basic parameters = "<<pointer_position;
      }
    
      //Write basic parameters to file
      size_t size = sizeof(mt_parameters[0]) * number_of_rounds;
      fwrite_status = fwrite(mt_parameters,size,1,outfile);
      //Update pointer position after writing
      pointer_position = ftell(outfile);
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfwrite status after writing basic parameters = "<<fwrite_status;
        cout<<"\npointer position after writing basic parameters = "<<pointer_position;
      }
    }
    
    else
    {
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\npointer position before reading mt_parameters = "<<pointer_position;
      }
      
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
        fseek_status = fseek(outfile,(pointer_position),SEEK_SET);
        pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\n\niteration = "<<iteration;
        cout<<"\nfseek status before reading basic parameters = "<<fseek_status;
        cout<<"\npointer position before reading basic parameters = "<<pointer_position;
      }
    
      //Write basic parameters to file
      size_t size = sizeof(mt_parameters[0]) * number_of_rounds;
      fread_status = fread(mt_parameters,size,1,outfile);
      //Update pointer position after writing
      pointer_position = ftell(outfile);
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfread status after reading basic parameters = "<<fread_status;
        cout<<"\npointer position after reading basic parameters = "<<pointer_position;
      } 
    }
    
    fclose(outfile);
    return pointer_position;
  }  
  
  
  static inline long rwIntegerArray(FILE *outfile,const char *file_path,const char *mode,uint32_t *map_choice_array,int length,long ptr_position)
  {
    int fwrite_status = 9,fseek_status = 9,fread_status = 9;
    long pointer_position = ptr_position;
    outfile = fopen(file_path,mode);
    
    if(outfile == NULL)
    {
      printf("\nCould not open parameters.bin for writing map choices array\nExiting...");
      exit(0);
    }
    
    if(strcmp("wb",mode) == 0 || strcmp("ab",mode) == 0)
    {
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\npointer position before writing map choices array = "<<ptr_position;
      }
      
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfseek status before writing map choices array = "<<fseek_status;
        cout<<"\npointer position after wriitng map choices array = "<<pointer_position;
      }
    
      size_t size = length * sizeof(map_choice_array[0]);
      fwrite_status = fwrite(map_choice_array,size,1,outfile);
      cout<<"\nfwrite status after writing map choices array = "<<fwrite_status;
    }
    
    else
    {
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\npointer position before reading map choices array = "<<ptr_position;
      }
      
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
        fseek_status = fseek(outfile,(pointer_position),SEEK_SET);
        pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfseek status before reading map choices array = "<<fseek_status;
        cout<<"\npointer position after reading map choices array = "<<pointer_position;
      }
    
      size_t size = length * sizeof(map_choice_array[0]);
      fread_status = fread(map_choice_array,size,1,outfile);
      cout<<"\nfread status after reading map choices array = "<<fread_status;
    }
    fclose(outfile);
    return pointer_position; 
  }
  
  
  void rwInteger(FILE *outfile,const char *file_path,const char *mode,int seed,int length,long ptr_position)
  {
    int fwrite_status = 9,fseek_status = 9,fread_status = 9;
    long pointer_position = ptr_position;
    outfile = fopen(file_path,mode);
    
    if(outfile == NULL)
    {
      printf("\nCould not open parameters.bin for writing integer\nExiting...");
      exit(0);
    }
    
    if(strcmp("wb",mode) == 0 || strcmp("ab",mode) == 0)
    {
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfseek status before writing integer = "<<fseek_status;
        cout<<"\npointer position before writing integer = "<<pointer_position;
      }
    
      size_t size = sizeof(seed);
      fwrite_status = fwrite(&seed,size,1,outfile);
      cout<<"\nfwrite status after writing integer = "<<fwrite_status;
      cout<<"\npointer position after writing integer = "<<pointer_position;
    }
    
    else
    {
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\npointer position before reading integer = "<<ptr_position;
      }
      
      //Offset pointer position by length of previous record
      if(pointer_position > 0)
      {
        fseek_status = fseek(outfile,(pointer_position),SEEK_SET);
        pointer_position = ftell(outfile);
      }
    
      if(DEBUG_READ_WRITE == 1)
      {
        cout<<"\nfseek status before reading integer = "<<fseek_status;
        cout<<"\npointer position before reading integer = "<<pointer_position;
      }
    
      size_t size = sizeof(seed);
      fread_status = fread(&seed,size,1,outfile);
      cout<<"\nfread status after reading integer = "<<fread_status;
      cout<<"\npointer position after reading integer = "<<pointer_position;
    }
    
    cout<<"\nClosed file in rwInteger";    
    fclose(outfile);
    //return pointer_position;  
  }

#endif

