  #include <iostream> /*For IO*/
  #include <cstdint>  /*For standardized data types*/
  #include "functions.hpp"
  #include "kernel.hpp"
  
  using namespace std;
  using namespace cv;
  
  int main()
  { 
    
    uint32_t M=0,N=0,MID=0,TOTAL=0,R_UPPER=0;
    uint8_t alpha1=0,beta1=0,a1=0,b1=0,offset1=0;
    uint8_t alpha2=0,beta2=0,a2=0,b2=0,offset2=0;
    
    cv::Mat image;
    
    image=cv::imread("lena.png",cv::IMREAD_GRAYSCALE);
    cv::resize(image,image,cv::Size(),0.01,0.01);
    
    M=(uint32_t)image.rows;
    N=(uint32_t)image.cols;
    
    cout<<"\nM= "<<M<<"\tN= "<<N;
    TOTAL=(M*N);
    MID=((M*N)/2);
    R_UPPER=(M*N)-N;
    


    uint16_t c1=0;
    uint16_t c2=0;
    
    int32_t r1=0;
    int32_t r2=0;
    
    float x1=0,y1=0,skipped_x1=0,skipped_y1=0; 
    float x2=0,y2=0,skipped_x2=0,skipped_y2=0;
    
    bool isNotDecimal=0;
    
    /*Declarations*/
    
    /*struct constants variables*/
    constants constants_array1;
    constants constants_array2;
    
    /*Miscellaneous Declarations*/
    std::vector<float> skipped_x_y1(NUM_SKIPOFFSET_ARGS);
    std::vector<float> skipped_x_y2(NUM_SKIPOFFSET_ARGS);
    std::vector<float> P1(TOTAL);
    std::vector<float> P2(TOTAL);
    std::vector<uint32_t> U(M);
    std::vector<uint32_t> V(M);
    std::vector<uint8_t> img_vec(TOTAL);
    std::vector<int32_t> retrieved_constants_array;    
    std::array<uint8_t,25> img_arr;
    std::array<uint32_t,5> U_arr;
    std::array<uint32_t,5> V_arr;
    
    /*GPU arrays*/
    
    uint8_t *gpuimgIn;
    uint8_t *gpuimgOut;
    uint32_t *gpuU;
    uint32_t *gpuV;
    
    /*To store content of corresponding vector*/
    uint8_t  *raw_img_vec=(uint8_t*)malloc(TOTAL*sizeof(uint8_t));
    uint32_t *raw_U=(uint32_t*)malloc(M*sizeof(uint32_t));
    uint32_t *raw_V=(uint32_t*)malloc(M*sizeof(uint32_t));
   
    //img_vec=flattenGrayScaleImage(image);
    
    
    //printImageContents(image);
    //cout<<"\n";
    //img_vec=flattenGrayScaleImage(image);
    /*printf("\nimg_vec=");
    
    for (int i=0;i<TOTAL;++i)
    {
      printf("%d\t",img_vec[i]);
    }*/
  


    /*First set of Constants*/
    constants_array1=initializeConstants(SEED1,4,R_UPPER);
    alpha1  = constants_array1.alpha;
    beta1   = constants_array1.beta;
    a1      = constants_array1.a;
    b1      = constants_array1.b;
    c1      = constants_array1.c;
    x1      = constants_array1.x;
    y1      = constants_array1.y;
    //r1    = constants_array1.r;
    r1      =1;
    offset1 = constants_array1.offset;
    skipped_x_y1=skipOffset(a1,b1,c1,x1,y1,UNZERO,offset1);
    skipped_x1=skipped_x_y1.at(0);
    skipped_y1=skipped_x_y1.at(1);
    
    printf("\na1=%d\tb1=%d\tc1=%d\tx1=%f\ty1=%f\tr1=%d\toffset1=%d\tskipped_x1=%f\tskipped_y1=%f\t",a1,b1,c1,x1,y1,r1,offset1,skipped_x1,skipped_y1); 
    
    writeToFile("constants1.txt",alpha1,beta1,a1,b1,c1,x1,y1,r1,offset1,WRITE_MULTIPLE_ARGS);
    generateP(P1,a1,b1,c1,skipped_x1,skipped_y1,UNZERO,MID,TOTAL);
    generateU(U,P1,r1,M,N);
    
    cout<<"\nP1=";
    for(int i=0;i<TOTAL;++i)
    {
      printf("%f\t",P1[i]);
    }
      
    cout<<"\nU=";
    for(int i=0;i<M;++i)
    {
      cout<<U[i]<<" ";
    }
    
    
    /*Second set of Constants*/
    constants_array2=initializeConstants(SEED2,6,R_UPPER);
    alpha2  = constants_array2.alpha;
    beta2   = constants_array2.beta;
    a2      = constants_array2.a;
    b2      = constants_array2.b;
    c2      = constants_array2.c;
    x2      = constants_array2.x;
    y2      = constants_array2.y;
    //r2    = constants_array2.r;
    r2      = 1;  
    offset2 = constants_array2.offset;
    skipped_x_y2=skipOffset(a2,b2,c2,x2,y2,UNZERO,offset2);
    skipped_x2=skipped_x_y2.at(0);
    skipped_y2=skipped_x_y2.at(1);
    writeToFile("constants2.txt",alpha2,beta2,a2,b2,c2,x2,y2,r2,offset2,WRITE_MULTIPLE_ARGS);
    
    printf("\na2=%d\tb2=%d\tc2=%d\tx2=%f\ty2=%f\tr2=%d\toffset2=%d\tskipped_x2=%f\tskipped_y2=%f\t",a2,b2,c2,x2,y2,r2,offset2,skipped_x2,skipped_y2);
    
    generateP(P2,a2,b2,c2,skipped_x2,skipped_y2,UNZERO,MID,TOTAL);
    generateU(V,P2,r2,N,M);
    
        
    cout<<"\nP2=";
    for(int i=0;i<TOTAL;++i)
    {
      printf("%f\t",P2[i]);
    }
      
    cout<<"\nV=";
    for(int i=0;i<M;++i)
    {
      cout<<V[i]<<" ";
    } 
   
    /*Copy img_vec to raw_img_vec*/
    for(int i=0;i<TOTAL;++i)
    {
      raw_img_vec[i]=img_vec[i];
    }
    
    
    for(int i=0;i<M;++i)
    {
      raw_U[i]=U[i];
      raw_V[i]=V[i];
    } 
    
    cudaMalloc((void**)&gpuimgIn,TOTAL*sizeof(uint8_t));
    cudaMalloc((void**)&gpuimgOut,TOTAL*sizeof(uint8_t));
    cudaMalloc((void**)&gpuU,M*sizeof(uint32_t));
    cudaMalloc((void**)&gpuV,M*sizeof(uint32_t));
    
    cudaMemcpy((void*)gpuimgIn,(const void*)raw_img_vec,200,cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpuU,(const void*)raw_U,40,cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpuV,(const void*)raw_V,40,cudaMemcpyHostToDevice);
    
    dim3 block(3,3,1);
    dim3 grid(M,N,1);    
    
    run_GenCatMap(gpuimgIn,gpuimgOut,gpuU,gpuV,grid,block);
    cudaMemcpy((void*)raw_img_vec,(const void*)gpuimgOut,200,cudaMemcpyDeviceToHost);
    
    cout<<"\nraw_img_vec after kernel=";
    for(int i=0;i<TOTAL;++i)
    {
      printf("%d\t",raw_img_vec[i]);
    } 
    
    return 0; 
  }
  

