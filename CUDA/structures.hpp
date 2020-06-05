#pragma once
#include <iostream>

using namespace std;

// Enumerator classes for configuration
enum class Chaos { Arnold, Logistic2Dv1, Logistic2Dv2 };
enum class Mode { ENC, DEC };

// Comntains paths
struct paths
{
    string file;
    string type;
    string fn_img;
    string fn_img_enc;
    string fn_img_dec;
}path;

// Contains getRandVec params
struct CRNG
{
    Chaos map;
    int core;
    double x;
    double y;
    unsigned short int offset;
}permuter[2];

// Contains details on no. of rounds
struct mainConfig
{
    uint8_t rounds = 4;
    CRNG diffuser[2];
    double diff_r;
}cfg;

void buildPaths(string file)
{
    path.file = "images/" + file;
    path.type = ".png";
    path.fn_img = path.file + path.type;
    path.fn_img_enc = path.file + "_ENC" + path.type;
    path.fn_img_dec = path.file + "_DEC" + path.type;
}

void showPermuter(CRNG tmp)
{
    cout << "Core: " << tmp.core << " : " << tmp.x << endl;
}
