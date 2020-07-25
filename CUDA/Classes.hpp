#ifndef CLASSES_H
#define CLASSES_H

using namespace std;

/*Contains all classes used in the program*/

enum class Chaos 
{ 
  ArnoldMap = 1, 
  twodLogisticMap,
  twodSineLogisticModulatedMap,
  twodLogisticAdjustedSineMap,
  twodLogisticAdjustedLogisticMap,
  MersenneTwister
};

enum class Mode 
{ 
  ENCRYPT = 1, 
  DECRYPT 
};

// Contains paths
class Paths
{
  public: 
    std::string file_path;
    std::string file_name;
    std::string type;
    std::string fn_img;
    std::string fn_img_enc;
    std::string fn_img_dec;
    
    inline std::string getFileNameFromPath(std::string filename)
    {
      const size_t last_slash_idx = filename.find_last_of("\\/");
      if (std::string::npos != last_slash_idx)
      {
        filename.erase(0, last_slash_idx + 1);
      }

      // Remove extension if present.
      const size_t period_idx = filename.rfind('.');
      if (std::string::npos != period_idx)
      {
        filename.erase(period_idx);
      }
      
      return filename;
    }

    
    inline void buildPaths(std::string file_path)
    {
      type = ".png";
      fn_img = file_path;
      file_name = getFileNameFromPath(file_path);
      fn_img_enc = file_name + "_ENC" + type;
      fn_img_dec = file_name + "_DEC" + type;
      
      std::cout<<"\nfile_path = "<<file_path;
      std::cout<<"\nfile_name = "<<file_name;
      std::cout<<"\ntype = "<<type;
      std::cout<<"\nfn_img = "<<fn_img;
      std::cout<<"\nfn_img_enc = "<<fn_img_enc;
      std::cout<<"\nfn_img_dec = "<<fn_img_dec<<"\n";
    }
};


// Contains getRandVec params
class Permuter
{
  public:
    Chaos map;
    double x;
    double y;
    double x_bar;
    double y_bar;
    double alpha;
    double beta;
    double myu;
    double r;
};

// Contains diffusion parameters
class Diffuser
{
  public:
    Chaos map;
    double x;
    double y;
    double x_bar;
    double y_bar;
    double alpha;
    double beta;
    double myu;
    double r;
};

class Protect
{
  uint32_t permute_protect;
  uint32_t diffuse_protect;
};

// Contains details on no. of rounds and number of rotations
class Configuration
{
  public:
    uint8_t rounds = 2;
    uint8_t rotations = 1;
};


class Random
{
  
  public:
     
      static inline int getRandomInteger(int LOWER_BOUND, int UPPER_BOUND)
      {
        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 seeder(seed);
        std::uniform_int_distribution<int> intGen(LOWER_BOUND, UPPER_BOUND);
        int alpha = intGen(seeder);
        return alpha;
      }
     
      static inline int getRandomUnsignedInteger8(int LOWER_BOUND, int UPPER_BOUND)
      {
        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 seeder(seed);
        std::uniform_int_distribution<uint8_t> intGen(LOWER_BOUND, UPPER_BOUND);
        auto randnum = intGen(seeder);
        return (uint8_t)randnum;
      }
     
     static inline double getRandomDouble(double LOWER_LIMIT, double UPPER_LIMIT)
     {
       std::random_device r;
       std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
       std::mt19937 seeder(seed);
       std::uniform_real_distribution<double> realGen(LOWER_LIMIT, UPPER_LIMIT);   
       auto randnum = realGen(seeder);
       return (double)randnum;
     }
    
     static inline Chaos crngAssigner(int LOWER_LIMIT, int UPPER_LIMIT)
     {
       Chaos chaotic_map;
       chaotic_map = (Chaos)getRandomInteger(LOWER_LIMIT, UPPER_LIMIT);
       return chaotic_map;
     }
};


/**
 * Class containing Mersenne Twister and Chaotic Maps. These are used as PRNGs 
 */
class CRNG
{
  public:
    double X;
    double Y;
    double X_BAR;
    double Y_BAR;
    
    int RANDOM_NUMBER;
    
    double R;
    double ALPHA;
    double BETA;
    double MYU; 
    Chaos Map; 
    
    CRNG(double x, double y, double x_bar, double y_bar, double alpha, double beta, double myu, double r, Chaos map)
    {
      X = x;
      Y = y;
      X_BAR = x_bar;
      Y_BAR = y_bar;
      R = r;
      ALPHA = alpha;
      BETA = beta;
      MYU = myu;
      Map = map;
      
      if(DEBUG_CONSTRUCTORS == 1)
      {
        std::cout<<"\nX = "<<X;
        std::cout<<"\nY = "<<Y;
        std::cout<<"\nX_BAR = "<<X_BAR;
        std::cout<<"\nY_BAR = "<<Y_BAR;
        std::cout<<"\nR = "<<R;
        std::cout<<"\nALPHA = "<<ALPHA;
        std::cout<<"\nBETA = "<<BETA;
        std::cout<<"\nMYU = "<<MYU;
        std::cout<<"\nMap = "<<int(Map);
      }                      
    }
  
   inline void twodLM(double &X, double &Y, const double &R)
   {
       X = R * (3 * Y + 1) * X * (1 - X);
       Y = R * (3 * X + 1) * Y * (1 - Y);
   }
   
   inline void ArnoldIteration(double &X, double &Y)
   {
     auto xtmp = X + Y;
     Y = X + 2 * Y;
     X = xtmp - (int)xtmp;
     Y = Y - (int)Y;
   }
   
   inline void twodSLMM(double &x, double &y, const double &alpha, const double &beta)
   {
    x = alpha * (sin(M_PI * y) + beta) * x * (1 - x);
    y = alpha * (sin(M_PI * x) + beta) * y * (1 - y);
    return; 
   }
  
   inline void twodLASM(double &x, double &y, const double &myu)
   {
     x = sin(M_PI * myu * (y + 3) * x * (1 - x));
     y = sin(M_PI * myu * (x + 3) * y * (1 - y));
     return;
   }
  
   inline void twodLALM(double &x, double &y, double &x_bar, double &y_bar, const double &myu)
   {
     x_bar = myu * (y * 3) * x * (1 - x);
     x = 4 * x_bar * (1 - x_bar);
     y_bar = myu * (x + 3) * y * (1 - y);
     y = 4 * y_bar * (1 - y_bar);
     return;
   }
   
   inline void CRNGUpdateHost(double &X, double &Y, double X_BAR, double Y_BAR, const double &ALPHA, const double &BETA, const double &MYU, const double &R, Chaos Map)
   {
     
     switch(Map)
     {
       
       case Chaos::ArnoldMap: 
       {
         //std::cout<<"\n Generalized Arnold Map selected";
         ArnoldIteration(X, Y);
       }
       break;
       
       case Chaos::twodLogisticMap: 
       {
         //std::cout<<"\n 2D Logistic Map selected";
         twodLM(X, Y, R);
       }
       break;
       
       case Chaos::twodSineLogisticModulatedMap: 
       {
         //std::cout<<"\n 2D Logistic Map selected";
         twodSLMM(X, Y, ALPHA, BETA);
       }
       break;
       
       case Chaos::twodLogisticAdjustedSineMap: 
       {
         //std::cout<<"\n 2D Logistic Map selected";
         twodLASM(X, Y, MYU);
       }
       break;
       
       case Chaos::twodLogisticAdjustedLogisticMap: 
       {
         //std::cout<<"\n 2D Logistic Map selected";
         twodLALM(X, Y, X_BAR, Y_BAR, MYU);
       }
       break;
       
       default: std::cout << "\nInvalid CRNG Option!\nExiting...";
       std::exit(0);
     }
   }
   
   
};

/*Class objects*/
Configuration config;
Paths path;
Permuter permute[2];
Diffuser diffuse;
//CRNG crng;
Random randomNumber;
Protect protect; 
#endif
