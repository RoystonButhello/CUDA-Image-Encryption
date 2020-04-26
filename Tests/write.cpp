#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
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
  cout<<"\nIN WRITE";
  read_s var[3];
  two_s tar[3];
  
  var[0].a = 1.011;
  var[0].b = 2;
  var[0].c = 3.812;
  
  var[1].a = 4.00;
  var[1].b = 5;
  var[1].c = 6.00;
  
  var[2].a = 7.01;
  var[2].b = 8;
  var[2].c = 9.10;
  
  tar[0].e = 10.0290;
  tar[0].f = 10.5124;

  tar[1].e = 11.0340;
  tar[1].f = 11.5218;

  tar[2].e = 12.0401;
  tar[2].f = 12.0910;
  
  int fseek_status = 9,fwrite_status = 0;
  long ptr_position = 0;
  
  printf("\nsizeof(var[0]) = %ld",sizeof(var[0]));
  printf("\nsizeof(tar[0]) = %ld",sizeof(tar[0]));
  
  /*Display var*/
  for(int i = 0; i < 3; ++i)
  {
    printf("\nvar.a = %f",var[i].a);
    printf("\nvar.b = %d",var[i].b);
    printf("\nvar.b = %f",var[i].c);
  }
  
  /*Display tar*/
  for(int i = 0; i < 3; ++i)
  {
    printf("\ntar.e = %f",tar[i].e);
    printf("\ntar.f = %f",tar[i].f);
  }
  
  int num1 = 10;
  int num2 = 20;
   
  FILE *outfile;
  /*Writing var*/
  for(int i = 0; i < 3; ++i)
  {
   
    if(i == 0)
    {
      outfile = fopen("s.bin","wb");
    }
    else
    {
       outfile = fopen("s.bin","ab");
    }
    
    if(outfile == NULL)
    {
      cout<<"\nFile not opened for writing\nExiting...";
      exit(0);
    }
    
    
    //Offset pointer position to 0 or by length of previous record 
    if(i == 0)
    {
      //Get pointer position before writing
      ptr_position = ftell(outfile);
      fseek_status = fseek(outfile,(ptr_position),SEEK_SET);
    }
    else
    {
       fseek_status = fseek(outfile,(ptr_position + 1),SEEK_SET);
    }
    
    //Write record to file
    fwrite_status = fwrite(&var[i],sizeof(var[i]),1,outfile);
    //Update pointer position after writing
    ptr_position = ftell(outfile);
    cout<<"\nfwrite status after writing var "<<i<<" = "<<fwrite_status;
    cout<<"\npointer position after writing var "<<i<<" = "<<ptr_position; 
    fclose(outfile);
  }
  
  /*Writing tar*/
  for(int i = 0; i < 3; ++i)
  {
    
    if(i == 0)
    {
      outfile = fopen("s.bin","ab");
    }
    else
    {
       outfile = fopen("s.bin","ab");
    }
    
    if(outfile == NULL)
    {
      cout<<"\nFile not opened for writing\nExiting...";
      exit(0);
    }
    
    
    //Offset pointer position by length of previous record 
    fseek_status = fseek(outfile,(ptr_position + 1),SEEK_SET);
    
    
    //Write record to file
    fwrite_status = fwrite(&tar[i],sizeof(tar[i]),1,outfile);
    //Update pointer position after writing
    ptr_position = ftell(outfile);
    cout<<"\nfwrite status after writing tar "<<i<<" = "<<fwrite_status;
    cout<<"\npointer position after writing tar "<<i<<" = "<<ptr_position; 
    fclose(outfile);
  }
  
  /*Writing num1*/ 
  outfile = fopen("s.bin","ab");
  
  if(outfile == NULL)
  {
    cout<<"\nFile not opened for writing\nExiting...";
    exit(0);
  }
  
  //Offset pointer position by length of previous record
  fseek_status = fseek(outfile,(ptr_position + 1),SEEK_SET);
  //Get offset pointer position
  ptr_position = ftell(outfile);
  
  cout<<"\nfseek status before writing num1 = "<<fseek_status;
  cout<<"\npointer position before writing num1 = "<<ptr_position;
  
  //Write record to file
  fwrite_status = fwrite(&num1,sizeof(num1),1,outfile);
  
  //Update pointer position after writing
  ptr_position = ftell(outfile);
  
  cout<<"\nfwrite status after writing num1 = "<<fwrite_status;
  cout<<"\npointer position after writing num = "<<ptr_position;
  fclose(outfile); 

  
  /*Writing num2*/ 
  outfile = fopen("s.bin","ab");
  
  if(outfile == NULL)
  {
    cout<<"\nFile not opened for writing\nExiting...";
    exit(0);
  }
  
  //Offset pointer position by length of previous record
  fseek_status = fseek(outfile,(ptr_position + 1),SEEK_SET);
  //Get offset pointer position
  ptr_position = ftell(outfile);
  
  cout<<"\nfseek status before writing num2 = "<<fseek_status;
  cout<<"\npointer position before writing num2 = "<<ptr_position;
  
  //Write record to file
  fwrite_status = fwrite(&num2,sizeof(num2),1,outfile);
  
  //Update pointer position after writing
  ptr_position = ftell(outfile);
  
  cout<<"\nfwrite status after writing num2 = "<<fwrite_status;
  cout<<"\npointer position after writing num2 = "<<ptr_position;
  fclose(outfile); 
  
  return 0;
}
