
// Driver code
#include "Core.hpp"

int main(int argc, char *argv[])
{
    if(strcmp(argv[1], "\0") == 0 || strcmp(argv[2], "\0") == 0)
    {
      cout<<"\nUse as:\n ./Main <absolute_file_path><a_single_space><number_of_rounds>\n";
      exit(0);
    }
    
    std::string file_path = std::string(argv[1]);
    
    
    std::string number_of_rounds = std::string(argv[2]);
    std::string rotations_per_round = std::string(argv[3]);
    int rounds = std::stoi(number_of_rounds);
    int rotations = std::stoi(rotations_per_round);
    
    Encrypt(file_path, rounds, rotations);
    Decrypt();
    
    
    return 0;
}
