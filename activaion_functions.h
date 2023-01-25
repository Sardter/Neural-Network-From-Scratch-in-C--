#ifndef __ACTIVATION
#define __ACTIVATION

#include "matrice_lib.h"
#include <functional>

double step_function(double x);

double step_function_derivative(double x);

double sigmoid_function(double x);

double sigmoid_function_derivative(double x);

double rectifeid_linear(double x);

double rectifeid_linear_derivative(double x);

matrice * apply_activation(matrice * data, function<double(double)> func);

matrice * soft_max_activation(matrice * data);

#endif