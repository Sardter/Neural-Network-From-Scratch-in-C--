#ifndef __LAYER_TOOLS
#define __LAYER_TOOLS

#include "matrice_lib.h"

using namespace std;

matrice * generate_rondom_weights(int rows, int columns);

vector<double> * default_biases(int neurons);

double calculate_accuracy(matrice * data, vector<int> * targets);

#endif