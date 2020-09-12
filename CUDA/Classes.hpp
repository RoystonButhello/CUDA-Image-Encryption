#ifndef CLASSES_H
#define CLASSES_H

using namespace std;

/*Contains all classes used in the program*/

/**
 * Enumeration class containing all PRNG choices
 */
enum class Chaos 
{ 
  ArnoldMap = 1, 
  twodLogisticMap,
  twodSineLogisticModulatedMap,
  twodLogisticAdjustedSineMap,
  twodLogisticAdjustedLogisticMap
};

/**
 * Enumeration class containing all modes of operation
 */
enum class Mode 
{ 
  ENCRYPT = 1, 
  DECRYPT 
};

/**
 * Class containing file paths and extensions
 */
class Paths
{
  public: 
    std::string file_path;
    std::string file_name;
    std::string type;
    std::string fn_img;
    std::string fn_img_enc;
    std::string fn_img_dec;
    
    /**
     * Gets file name from absolute file path. Takes absolute image file path as argument
     */
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

    /**
     * Initializes file paths at the beginning of operation. Takes absolute image file path as argument
     */
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


/**
 * Class containing parameters of all chaotic maps used in generating permutation vectors 
 */
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

/**
 * Class containing parameters of all chaotic maps used in generating diffusion vectors 
 */
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

/**
 * Class containing permutation and diffusion parameter modifiers and offsets used to modify permutation and diffusion parameters
 */
class Offset
{
  public:
    double permute_param_modifier;
    double diffuse_param_modifier;
    double permute_param_offset;
    double diffuse_param_offset;
};

/** 
 *  Class containing details on number of encryption and decryption rounds and number of permutations (rotations) per round
 */
class Configuration
{
  public:
    uint8_t rounds;
    uint8_t rotations;
};

/**
 * Class containing functions for generating random numbers using MT-19937 PRNG
 */
class Random
{
  
  public:
     
      /**
       * Returns a random 32-bit signed integer in the range of (LOWER_BOUND, UPPER_BOUND). Takes the lower bound and the upper bound as arguments
       */
      static inline int getRandomInteger(int LOWER_BOUND, int UPPER_BOUND)
      {
        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 seeder(seed);
        std::uniform_int_distribution<int> intGen(LOWER_BOUND, UPPER_BOUND);
        int alpha = intGen(seeder);
        return alpha;
      }
     
      /**
       * Returns a random 8-bit unsigned integer in the range of (LOWER_BOUND, UPPER_BOUND). Takes the lower bound and the upper bound as arguments
       */
      static inline int getRandomUnsignedInteger8(int LOWER_BOUND, int UPPER_BOUND)
      {
        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 seeder(seed);
        std::uniform_int_distribution<uint8_t> intGen(LOWER_BOUND, UPPER_BOUND);
        auto randnum = intGen(seeder);
        return (uint8_t)randnum;
      }
     
     /**
      * Returns a random 8-bit unsigned integer in the range of (LOWER_BOUND, UPPER_BOUND). Takes the lower bound and the upper bound as arguments
      */
      static inline int getRandomUnsignedInteger32(int LOWER_BOUND, int UPPER_BOUND)
      {
        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 seeder(seed);
        std::uniform_int_distribution<uint32_t> intGen(LOWER_BOUND, UPPER_BOUND);
        auto randnum = intGen(seeder);
        return (uint32_t)randnum;
      }

    
     /**
      * Returns a random 32-bit double in the range of (LOWER_LIMIT, UPPER_LIMIT). Takes the lower limit and the upper limit as arguments
      */
     static inline double getRandomDouble(double LOWER_LIMIT, double UPPER_LIMIT)
     {
       std::random_device r;
       std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
       std::mt19937 seeder(seed);
       std::uniform_real_distribution<double> realGen(LOWER_LIMIT, UPPER_LIMIT);   
       auto randnum = realGen(seeder);
       return (double)randnum;
     }
    
     /**
      * Returns a random instance of Chaos in the range of (LOWER_LIMIT, UPPER_LIMIT). Takes the lower limit and the upper limit as arguments
      */
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
  
   /**
    * 2D Logistic Map. Takes initializing parameters X, Y and control parameter R as arguments
    */
   inline void twodLM(double &X, double &Y, const double &R)
   {
       X = R * (3 * Y + 1) * X * (1 - X);
       Y = R * (3 * X + 1) * Y * (1 - Y);
   }
   
   /**
    * Arnold Map. Takes initializing parameters X and Y as arguments 
    */
   inline void ArnoldIteration(double &X, double &Y)
   {
     auto xtmp = X + Y;
     Y = X + 2 * Y;
     X = xtmp - (int)xtmp;
     Y = Y - (int)Y;
   }
   
   /**
    * 2D Sine Logistic Modulation Map. Takes initializing parameters X, Y and control parameters alpha and beta as arguments 
    */
   inline void twodSLMM(double &x, double &y, const double &alpha, const double &beta)
   {
    x = alpha * (sin(M_PI * y) + beta) * x * (1 - x);
    y = alpha * (sin(M_PI * x) + beta) * y * (1 - y);
    return; 
   }
  
   /**
    * 2D Sine Logistic Modulation Map. Takes initializing parameters X, Y and control parameter myu arguments 
    */
   inline void twodLASM(double &x, double &y, const double &myu)
   {
     x = sin(M_PI * myu * (y + 3) * x * (1 - x));
     y = sin(M_PI * myu * (x + 3) * y * (1 - y));
     return;
   }
   
   /**
    * 2D Sine Logistic Adjusted Logistic Map. Takes initializing parameters x,y,x_bar,y_bar and control parameter myu as arguments 
    */
   inline void twodLALM(double &x, double &y, double &x_bar, double &y_bar, const double &myu)
   {
     x_bar = myu * (y * 3) * x * (1 - x);
     x = 4 * x_bar * (1 - x_bar);
     y_bar = myu * (x + 3) * y * (1 - y);
     y = 4 * y_bar * (1 - y_bar);
     return;
   }
   
   /**
    * CRNG host update function used to select chaotic map for generating permutation or diffusion vectors. Takes all initializing parameters and contro parameters of all CRNGs included in Permute and Diffuse classes  
    */
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

/**
 * Class containing propagation factors used for Forward change propagation during permutation and diffusion
 */
class Propagation
{
  public:
    uint32_t permute_propagation_factor;
    uint32_t diffuse_propagation_factor;
};

/*Class objects*/
Configuration config;
Paths path;
Permuter permute[2];
Diffuser diffuse;
//CRNG crng;
Random randomNumber;
Offset offset;
Propagation propagator; 
#endif
