#include "matrice_lib.h"
#include <cmath>
#include <stdexcept>

using namespace std;

double clip(double x) {
    if (x < 1e-7) return 1e-7;
    else if (x > 1 - 1e-7) return 1 - 1e-7;
    return x;
}

double categorical_cross_entropy(matrice * input, vector<int> * target) {
    size_t col = input->column_size();
    size_t row = input->row_size();
    
    double sum = 0;
    size_t i;
    for (i = 0; i < col; i++)
    {
        sum += -log(clip(input->data[i][target->at(i)]));
    }
    
    return sum / col;
}