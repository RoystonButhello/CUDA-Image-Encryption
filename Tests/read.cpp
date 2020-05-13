#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
using namespace std;

typedef struct
{
  double a;
  int b;
  double c; 
}read_s;  

typedef struct
{
  double e;
  double f;
}two_s;

int main()
{
  read_s var[3];
  two_s  tar[3];

  int fseek_status = 9,fread_status = 0;
  long ptr_position = 0;
  
  printf("\n\nIN READ");
  printf("\nsizeof(var[0]) = %ld",sizeof(var[0]));
  printf("\nsizeof(tar[0]) = %ld",sizeof(tar[0]));
  
  int num1 = 0;
  int num2 = 0;
  
  FILE *infile;
  /*Reading var*/
  for(int i = 0; i < 3; ++i)
  {
    infile = fopen("s.bin","rb");
    if(infile == NULL)
    {
      cout<<"\nCouldn't open file for reading";
    }
      
    //Offset position of file pointer by length of previous record
    fseek_status = fseek(infile,(ptr_position),SEEK_SET);
    //Get this offet position of file pointer
    ptr_position = ftell(infile);    

    //Read record from file
    fread_status = fread(&var[i],sizeof(var[i]),1,infile);
  
    printf("\nvar.a = %f ",var[i].a);
    printf("\nvar.b = %d ",var[i].b);
    printf("\nvar.c = %f ",var[i].c);
  
    //Update position of file pointer after reading
    ptr_position = ftell(infile);
    printf("\nfread_status after reading var = %d ",fread_status);
    cout<<"\npointer position after reading var "<<i<<"= "<<ptr_position;
    fclose(infile);
  }

  
  
  /*Read tar*/
  for(int i = 0; i < 3; ++i)
  {
    infile = fopen("s.bin","rb");
    if(infile == NULL)
    {
      cout<<"\nCouldn't open file for reading\nExiting...";
      exit(0);
    } 
    
    //Offset position of file pointer by length of previous record
    fseek_status = fseek(infile,(ptr_position),SEEK_SET);
    ptr_position = ftell(infile);
  
    cout<<"\nfseek status before reading tar "<<i<<" = "<<fseek_status;
    cout<<"\npointer position before reading tar "<<i<<" = "<<ptr_position;
    
    //Read record from file
    fread_status = fread(&tar[i],sizeof(tar[i]),1,infile);
    printf("\ntar.e = %f",tar[i].e);
    printf("\ntar.f = %f",tar[i].f);
    //Update position of file pointer after reading
    ptr_position = ftell(infile);
  
    cout<<"\nfread status after reading tar "<<i<<" = "<<fread_status;
    cout<<"\npointer position after reading "<<i<<" = "<<ptr_position;
    fclose(infile);
  }
  
  /*Reading num1*/ 
  infile = fopen("s.bin","rb");
  
  if(infile == NULL)
  {
    cout<<"\nFile not opened for reading\nExiting...";
    exit(0);
  }
  
  //Offset pointer position by length of previous record
  fseek_status = fseek(infile,(ptr_position),SEEK_SET);
  //Get offset pointer position
  ptr_position = ftell(infile);
  
  cout<<"\nfseek status before reading num1 = "<<fseek_status;
  cout<<"\npointer position before reading num1 = "<<ptr_position;
  
  //Read record from file
  fread_status = fread(&num1,sizeof(num1),1,infile);
  cout<<"\nnum1 = "<<num1;
  //Update pointer position after writing
  ptr_position = ftell(infile);
  
  cout<<"\nfread status after reading num1 = "<<fread_status;
  cout<<"\npointer position after writing num = "<<ptr_position;
  fclose(infile); 
  
  /*Reading num2*/ 
  infile = fopen("s.bin","rb");
  
  if(infile == NULL)
  {
    cout<<"\nFile not opened for reading\nExiting...";
    exit(0);
  }
  
  //Offset pointer position by length of previous record
  fseek_status = fseek(infile,(ptr_position),SEEK_SET);
  //Get offset pointer position
  ptr_position = ftell(infile);
  
  cout<<"\nfseek status before reading num2 = "<<fseek_status;
  cout<<"\npointer position before reading num2 = "<<ptr_position;
  
  //Read record from file
  fread_status = fread(&num2,sizeof(num2),1,infile);
  cout<<"\nnum2 = "<<num2;
  //Update pointer position after writing
  ptr_position = ftell(infile);
  
  cout<<"\nfread status after reading num2 = "<<fread_status;
  cout<<"\npointer position after writing num2 = "<<ptr_position;
  fclose(infile); 
  
  return 0;
}
