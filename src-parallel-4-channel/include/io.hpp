#ifndef IO_H
#define IO_H

  static inline long writeLMParameters(FILE *outfile,const char *file_path,const char *write_mode,config::lm lm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long writeLMAParameters(FILE *outfile,const char *file_path,const char *write_mode,config::lma lma_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long writeSLMMParameters(FILE *outfile,const char *file_path,const char *write_mode,config::slmm slmm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long writeLASMParameters(FILE *outfile,const char *file_path,const char *write_mode,config::lasm lasm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long writeLALMParameters(FILE *outfile,const char *file_path,const char *write_mode,config::lalm lalm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long writeMTParameters(FILE *outfile,const char *file_path,const char *write_mode,config::mt mt_parameters[],int iteration,int number_of_rounds,long ptr_position);
  
  

  static inline long readLMParameters(FILE *infile,const char *file_path,const char *read_mode,config::lm lm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long readLMAParameters(FILE *infile,const char *file_path,const char *read_mode,config::lma lma_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long readSLMMParameters(FILE *infile,const char *file_path,const char *read_mode,config::slmm slmm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long readLASMParameters(FILE *infile,const char *file_path,const char *read_mode,config::lasm lasm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long readSLMMParameters(FILE *infile,const char *file_path,const char *read_mode,config::slmm slmm_parameters[],int iteration,int number_of_rounds,long ptr_position);
  static inline long readMTParameters(FILE *infile,const char *file_path,const char *read_mode,config::mt mt_parameters[],int iteration,int number_of_rounds,long ptr_position);
  

static inline long writeLMParameters(FILE *outfile,const char *file_path,const char *write_mode,config::lm lm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    long pointer_position = ptr_position;
    int fwrite_status = 9,fseek_status = 9;
    outfile = fopen(file_path,write_mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    
    //Offset pointer position by length of previous record
    if(pointer_position > 0)
    {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before writing lm_parameters = "<<fseek_status;
    cout<<"\npointer position before writing lm_parameters = "<<pointer_position;
    //Write lm parameters to file
    size_t size = number_of_rounds * sizeof(lm_parameters[0]);
    fwrite_status = fwrite(lm_parameters,size,1,outfile);
    //Update pointer position after writing
    pointer_position = ftell(outfile);
    cout<<"\nfwrite status after writing lm_parameters = "<<fwrite_status;
    cout<<"\npointer position after writing lm_parameters = "<<pointer_position;
    fclose(outfile);
    return pointer_position;
  }
  
  static inline long writeLMAParameters(FILE *outfile,const char *file_path,const char *write_mode,config::lma lma_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    long pointer_position = ptr_position;
    int fwrite_status = 9,fseek_status = 9;
    outfile = fopen(file_path,write_mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    
    //Offset pointer position by length of previous record
    if(pointer_position > 0)
    {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before writing lma_parameters = "<<fseek_status;
    cout<<"\npointer position before writing lma_parameters = "<<pointer_position;
    //Write lma parameters to file
    size_t size = sizeof(lma_parameters[0]) * number_of_rounds;
    fwrite_status = fwrite(lma_parameters,size,1,outfile);
    //Update pointer position after writing
    pointer_position = ftell(outfile);
    cout<<"\nfwrite status after writing lma_parameters = "<<fwrite_status;
    cout<<"\npointer position after writing lma_parameters = "<<pointer_position;
    fclose(outfile);
    return pointer_position;
  }
  
  
  static inline long writeSLMMParameters(FILE *outfile,const char *file_path,const char *write_mode,config::slmm slmm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    long pointer_position = ptr_position;
    int fwrite_status = 9,fseek_status = 9;
    outfile = fopen(file_path,write_mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    
    //Offset pointer position by length of previous record
    if(pointer_position > 0)
    {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before writing slmm_parameters = "<<fseek_status;
    cout<<"\npointer position before writing slmm_parameters = "<<pointer_position;
    //Write slmm parameters to file
    size_t size = sizeof(slmm_parameters[0]) * number_of_rounds;
    fwrite_status = fwrite(slmm_parameters,size,1,outfile);
    //Update pointer position after writing
    pointer_position = ftell(outfile);
    cout<<"\nfwrite status after writing slmm_parameters = "<<fwrite_status;
    cout<<"\npointer position after writing slmm_parameters = "<<pointer_position;
    fclose(outfile);
    return pointer_position;
  }
  
  static inline long writeLASMParameters(FILE *outfile,const char *file_path,const char *write_mode,config::lasm lasm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    long pointer_position = ptr_position;
    int fwrite_status = 9,fseek_status = 9;
    outfile = fopen(file_path,write_mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    
    //Offset pointer position by length of previous record
    if(pointer_position > 0)
    {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before writing lasm_parameters = "<<fseek_status;
    cout<<"\npointer position before writing lasm_parameters = "<<pointer_position;
    //Write lasm parameters to file
    size_t size = sizeof(lasm_parameters[0]) * number_of_rounds;
    fwrite_status = fwrite(lasm_parameters,size,1,outfile);
    //Update pointer position after writing
    pointer_position = ftell(outfile);
    cout<<"\nfwrite status after writing lasm_parameters = "<<fwrite_status;
    cout<<"\npointer position after writing lasm_parameters = "<<pointer_position;
    fclose(outfile);
    return pointer_position;
  } 
  
  static inline long writeLALMParameters(FILE *outfile,const char *file_path,const char *write_mode,config::lalm lalm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    long pointer_position = ptr_position;
    int fwrite_status = 9,fseek_status = 9;
    outfile = fopen(file_path,write_mode);
    
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    
    //Offset pointer position by length of previous record
    if(pointer_position > 0)
    {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before writing lalm_parameters = "<<fseek_status;
    cout<<"\npointer position before writing lalm_parameters = "<<pointer_position;
    //Write lalm parameters to file
    size_t size = sizeof(lalm_parameters[0]) * number_of_rounds;
    fwrite_status = fwrite(lalm_parameters,size,1,outfile);
    //Update pointer position after writing
    pointer_position = ftell(outfile);
    cout<<"\nfwrite status after writing lalm_parameters = "<<fwrite_status;
    cout<<"\npointer position after writing lalm_parameters = "<<pointer_position;
    fclose(outfile);
    return pointer_position;
  }
  
  static inline long writeMTParameters(FILE *outfile,const char *file_path,const char *write_mode,config::mt mt_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    int fseek_status = 9,fwrite_status = 9;
    long pointer_position = ptr_position;
    outfile = fopen(file_path,write_mode);
    if(outfile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"\nExiting...";
      exit(0);
    }
    
    //Offset pointer position by length of previous record
    if(pointer_position > 0)
    {
        fseek_status = fseek(outfile,(pointer_position + 1),SEEK_SET);
        pointer_position = ftell(outfile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before writing basic parameters = "<<fseek_status;
    cout<<"\npointer position before writing basic parameters = "<<pointer_position;
    //Write basic parameters to file
    size_t size = sizeof(mt_parameters[0]) * number_of_rounds;
    fwrite_status = fwrite(mt_parameters,size,1,outfile);
    //Update pointer position after writing
    pointer_position = ftell(outfile);
    cout<<"\nfwrite status after writing basic parameters = "<<fwrite_status;
    cout<<"\npointer position after writing basic parameters = "<<pointer_position;
    fclose(outfile);
    return pointer_position;
    
  }  
  
  
  static inline long readLMParameters(FILE *infile,const char *file_path,const char *read_mode,config::lm lm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    
    int fread_status = 9,fseek_status = 9;
    long pointer_position = ptr_position;
    infile = fopen(file_path,read_mode); 
    
    if(infile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"for reading\nExiting...";
      exit(0);
    }
    
    
    if(pointer_position > 0)
    {
      //Offset pointer position by length of previous record
      fseek_status = fseek(infile,(pointer_position),SEEK_SET);
      //Get offset pointer position
      pointer_position = ftell(infile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before reading lm_parameters = "<<fseek_status;
    cout<<"\npointer position before reading lm_parameters = "<<pointer_position;
    //Read lm_parameters from file
    size_t size = number_of_rounds * sizeof(lm_parameters[0]);
    fread_status = fread(lm_parameters,size,1,infile);
    //Update pointer position after reading
    pointer_position = ftell(infile);
    cout<<"\nfread status after reading lm_parameters = "<<fread_status;
    cout<<"\npointer position after reading lm_parameters = "<<pointer_position;
    return pointer_position;
  }
  
  static inline long readLMAParameters(FILE *infile,const char *file_path,const char *read_mode,config::lma lma_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    
    int fread_status = 9,fseek_status = 9;
    long pointer_position = ptr_position;
    infile = fopen(file_path,read_mode); 
    
    if(infile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"for reading\nExiting...";
      exit(0);
    }
    
    
    if(pointer_position > 0)
    {
      //Offset pointer position by length of previous record
      fseek_status = fseek(infile,(pointer_position),SEEK_SET);
      //Get offset pointer position
      pointer_position = ftell(infile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before reading lma_parameters = "<<fseek_status;
    cout<<"\npointer position before reading lma_parameters = "<<pointer_position;
    //Read lm_parameters from file
    size_t size = number_of_rounds * sizeof(lma_parameters[0]);
    fread_status = fread(lma_parameters,size,1,infile);
    //Update pointer position after reading
    pointer_position = ftell(infile);
    cout<<"\nfread status after reading lma_parameters = "<<fread_status;
    cout<<"\npointer position after reading lma_parameters = "<<pointer_position;
    return pointer_position;
  }
  
  static inline long readSLMMParameters(FILE *infile,const char *file_path,const char *read_mode,config::slmm slmm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    
    int fread_status = 9,fseek_status = 9;
    long pointer_position = ptr_position;
    infile = fopen(file_path,read_mode); 
    
    if(infile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"for reading\nExiting...";
      exit(0);
    }
    
    
    if(pointer_position > 0)
    {
      //Offset pointer position by length of previous record
      fseek_status = fseek(infile,(pointer_position),SEEK_SET);
      //Get offset pointer position
      pointer_position = ftell(infile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before reading slmm_parameters = "<<fseek_status;
    cout<<"\npointer position before reading slmm_parameters = "<<pointer_position;
    //Read slmm_parameters from file
    size_t size = number_of_rounds * sizeof(slmm_parameters[0]);
    fread_status = fread(slmm_parameters,size,1,infile);
    //Update pointer position after reading
    pointer_position = ftell(infile);
    cout<<"\nfread status after reading slmm_parameters = "<<fread_status;
    cout<<"\npointer position after reading slmm_parameters = "<<pointer_position;
    return pointer_position;
  }
  
  static inline long readLASMParameters(FILE *infile,const char *file_path,const char *read_mode,config::lasm lasm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    
    int fread_status = 9,fseek_status = 9;
    long pointer_position = ptr_position;
    infile = fopen(file_path,read_mode); 
    
    if(infile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"for reading\nExiting...";
      exit(0);
    }
    
    
    if(pointer_position > 0)
    {
      //Offset pointer position by length of previous record
      fseek_status = fseek(infile,(pointer_position),SEEK_SET);
      //Get offset pointer position
      pointer_position = ftell(infile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before reading lasm_parameters = "<<fseek_status;
    cout<<"\npointer position before reading lasm_parameters = "<<pointer_position;
    //Read lasm_parameters from file
    size_t size = number_of_rounds * sizeof(lasm_parameters[0]);
    fread_status = fread(lasm_parameters,size,1,infile);
    //Update pointer position after reading
    pointer_position = ftell(infile);
    cout<<"\nfread status after reading lasm_parameters = "<<fread_status;
    cout<<"\npointer position after reading lasm_parameters = "<<pointer_position;
    return pointer_position;
  }
  
  static inline long readLALMParameters(FILE *infile,const char *file_path,const char *read_mode,config::lalm lalm_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    
    int fread_status = 9,fseek_status = 9;
    long pointer_position = ptr_position;
    infile = fopen(file_path,read_mode); 
    
    if(infile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"for reading\nExiting...";
      exit(0);
    }
    
    
    if(pointer_position > 0)
    {
      //Offset pointer position by length of previous record
      fseek_status = fseek(infile,(pointer_position),SEEK_SET);
      //Get offset pointer position
      pointer_position = ftell(infile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before reading lalm_parameters = "<<fseek_status;
    cout<<"\npointer position before reading lalm_parameters = "<<pointer_position;
    //Read lasm_parameters from file
    size_t size = number_of_rounds * sizeof(lalm_parameters[0]);
    fread_status = fread(lalm_parameters,size,1,infile);
    //Update pointer position after reading
    pointer_position = ftell(infile);
    cout<<"\nfread status after reading lasm_parameters = "<<fread_status;
    cout<<"\npointer position after reading lasm_parameters = "<<pointer_position;
    return pointer_position;
  }
  
  static inline long readMTParameters(FILE *infile,const char *file_path,const char *read_mode,config::mt mt_parameters[],int iteration,int number_of_rounds,long ptr_position)
  {
    
    int fread_status = 9,fseek_status = 9;
    long pointer_position = ptr_position;
    infile = fopen(file_path,read_mode); 
    
    if(infile == NULL)
    {
      cout<<"\nCould not open "<<file_path<<"for reading\nExiting...";
      exit(0);
    }
    
    
    if(pointer_position > 0)
    {
      //Offset pointer position by length of previous record
      fseek_status = fseek(infile,(pointer_position),SEEK_SET);
      //Get offset pointer position
      pointer_position = ftell(infile);
    }
    
    cout<<"\n\niteration = "<<iteration;
    cout<<"\nfseek status before reading basic parameters = "<<fseek_status;
    cout<<"\npointer position before reading basic parameters = "<<pointer_position;
    //Read basic parameters from file
    size_t size = number_of_rounds * sizeof(mt_parameters[0]);
    fread_status = fread(mt_parameters,size,1,infile);
    //Update pointer position after reading
    pointer_position = ftell(infile);
    cout<<"\nfread status after reading basic parameters = "<<fread_status;
    cout<<"\npointer position after reading basic parameters = "<<pointer_position;
    return pointer_position;
  } 
  

#endif

