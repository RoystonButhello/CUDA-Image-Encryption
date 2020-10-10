/*Contains all classes used in the program*/

#ifndef CLASSES_H
#define CLASSES_H

#include <iostream>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  //define something for Windows (32-bit and 64-bit, this part is common)
  #include <corecrt_math_defines.h> // For M_PI
  
  #ifdef _WIN64
    //define something for Windows (64-bit only)
    #include <corecrt_math_defines.h> // For M_PI
  #else
    //define something for Windows (32-bit only)
    #include <corecrt_math_defines.h> // For M_PI
  #endif

#elif __linux__
  // Linux
  #include <cmath> // For M_PI
  
#elif __unix__ // all unices not caught above
  // Unix
  #include <cmath> // For M_PI
  
#elif defined(_POSIX_VERSION)
  // POSIX
  #include <cmath> // For M_PI

#else
  #error "Unknown compiler"
#endif

using namespace std;

// PRNG choices
enum class Chaos
{
    Arnold,
    LM,
    SLMM,
    LASM,
    LALM
};

// Opmodes
enum class Mode:bool
{
    ENC = true,
    DEC = false
};

// File paths
class Paths
{
    public:
        string file_name;
        string file_type;
        string fn_img;
        string fn_img_enc;
        string fn_img_dec;

        // Initializes paths at the beginning of operation
        inline void buildPaths(string file)
        {
            file_name = "images/" + file;
            file_type = ".png";
            fn_img = file_name + file_type;
            fn_img_enc = fn_img + "_ENC" + file_type;
            fn_img_dec = fn_img + "_DEC" + file_type;
        }
};

// Number of rounds
class Configuration
{
    public:
        uint8_t rounds;
        uint8_t rotations;
};

// Parameters of all chaotic maps
class CRNG
{
    public:
        Chaos map = Chaos::Arnold;
        double x = 0.0;
        double y = 0.0;
        double alpha = 0.0;
        double beta = 0.0;
        double myu = 0.0;
        double r = 0.0;
};

// Parameter modifiers and offsets
class Offset
{
    public:
        double perm_modifier;
        double diff_modifier;
        double perm_offset;
        double diff_offset;
};

class Propagation
{
    public:
    uint32_t perm_propfac;
    uint32_t diff_propfac;
};

// Object creation
Configuration config;
Paths path;
CRNG permute[2];
CRNG diffuse;
Offset offset;
Propagation prop;

#endif
