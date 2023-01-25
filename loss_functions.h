#ifndef __LOSS
#define __LOSS

#include "matrice_lib.h"
#include <vector>

using namespace std;

double categorical_cross_entropy(matrice * input, vector<int> * target);

#endif