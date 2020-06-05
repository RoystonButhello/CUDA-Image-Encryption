                                        ==================BREAKUP OF FILES===========================

include - Contains all program header files.
main.cpp - Contains image encryption algorithm.
decrypt.cpp - Contains image decryption algorithm.
kernel - Contains CUDA kernel.
docs - Contains all software documentation.

NOTE: In the docs folder, documentation is provided in two forms: A non-interactive form, as a single PDF file, and an interactive form, as in the html folder. To use the html documentation, open the index.html file. It is the home-page of the interactive documentation and has links to all other pages.

To use the GUI, simply click on the Select Image and Run Button. A dialog box will appear, it will ask you to select the image. On selecting the image, the encryption and decryption processes will run automatically. THis will be followed by another dialog box which will ask you to select the location of the run.sh script. Once both the image and the script have been selected, then the encryption and decryption programs will be compiled and executed. 

To configure the algorithm to run using different modes, use the config.hpp header file and set the flags to either 1 or 0 to enable or disable a feature, respectively.
















 





 
