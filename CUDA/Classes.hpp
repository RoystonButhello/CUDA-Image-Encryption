#ifndef CLASSES_H
#define CLASSES_H

#include <iostream>
#include <corecrt_math_defines.h> // For M_PI

using namespace std;

/*Contains all classes used in the program*/

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
        std::string file_name;
        std::string file_type;
        std::string fn_img;
        std::string fn_img_enc;
        std::string fn_img_dec;

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
        Chaos map;
        double x;
        double y;
        double alpha;
        double beta;
        double myu;
        double r;
};

// Parameter modifiers and offsets
class Offset
{
    public:
        double permute_param_modifier;
        double diffuse_param_modifier;
        double permute_param_offset;
        double diffuse_param_offset;
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
Propagation propagator;

#endif
