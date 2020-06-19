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
    std::string file;
    std::string type;
    std::string fn_img;
    std::string fn_img_enc;
    std::string fn_img_dec;
    
    inline void buildPaths(std::string file)
    {
      file = "images/" + file;
      type = ".png";
      fn_img = file + type;
      fn_img_enc = file + "_ENC" + type;
      fn_img_dec = file + "_DEC" + type;
      
      std::cout<<"\nfile = "<<file;
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
    int mt_seed;
};

class Diffuser
{
  public:
    Chaos map;
    double x;
    double y;
    double r;
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
    
    int LOWER_LIMIT;
    int UPPER_LIMIT;
    
    int RANDOM_NUMBER;
    
    double R;
    double ALPHA;
    double BETA;
    double MYU;
    int MT_SEED; 
    Chaos Map; 
    
    CRNG(double x, double y, double x_bar, double y_bar, int lower_limit, int upper_limit, double alpha, double beta, double myu, double r, int mt_seed, Chaos map)
    {
      X = x;
      Y = y;
      X_BAR = x_bar;
      Y_BAR = y_bar;
      LOWER_LIMIT = lower_limit;
      UPPER_LIMIT = upper_limit;
      R = r;
      ALPHA = alpha;
      BETA = beta;
      MYU = myu;
      MT_SEED = mt_seed;
      Map = map;
      
      if(DEBUG_CONSTRUCTORS == 1)
      {
        std::cout<<"\nX = "<<X;
        std::cout<<"\nY = "<<Y;
        std::cout<<"\nX_BAR = "<<X_BAR;
        std::cout<<"\nY_BAR = "<<Y_BAR;
        std::cout<<"\nLOWER_LIMIT = "<<LOWER_LIMIT;
        std::cout<<"\nUPPER_LIMIT = "<<UPPER_LIMIT;
        std::cout<<"\nR = "<<R;
        std::cout<<"\nALPHA = "<<ALPHA;
        std::cout<<"\nBETA = "<<BETA;
        std::cout<<"\nMYU = "<<MYU;
        std::cout<<"\nMT_SEED = "<<MT_SEED;
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
   
   inline void Logistic2Dv2Iteration(double& x, double& y, const std::vector<double> &v)
   {
    auto xtmp = x;
    x = x * v[0] * (1 - x) + v[2] * y * y;
    y = y * v[1] * (1 - y) + v[3] * xtmp * (xtmp + y);
    return;
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
   
   inline void MT(int lower_limit, int upper_limit, int mt_seed, int &random_number)
   {
         std::mt19937 seeder(mt_seed);
         std::uniform_int_distribution<int> intGen(lower_limit, upper_limit);
         random_number = (int)intGen(seeder);
         return;
   }
   
   inline void CRNGUpdateHost(double &X, double &Y, double X_BAR, double Y_BAR, int LOWER_LIMIT, int UPPER_LIMIT, const double &ALPHA, const double &BETA, const double &MYU, const double &R, const double MT_SEED, Chaos Map)
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
       
       case Chaos::MersenneTwister:
       {
         MT(LOWER_LIMIT, UPPER_LIMIT, MT_SEED, RANDOM_NUMBER);
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
#endif
