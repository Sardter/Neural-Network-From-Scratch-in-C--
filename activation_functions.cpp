#include "activaion_functions.h"
#include <cstdlib>
#include <functional>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

double step_function(double x) {
    return x > 0;
}

double step_function_derivative(double x) {
    return 0;
}

double sigmoid_function(double x) {
    return x / (1 + abs(x));
}

double sigmoid_function_derivative(double x) {
    double e = exp(x);

    return e / ((1 + e) * (1 + e));
}

double rectifeid_linear(double x) {
    return x > 0 ? x : 0;
}

double rectifeid_linear_derivative(double x) {
    return x > 0;
}

matrice * apply_activation(matrice * data, function<double(double)> func) {
    for (size_t i = 0; i < data->column_size(); i++)
    {
        for (size_t j = 0; j < data->row_size(); j++)
        {
            data->data[i][j] = func(data->data[i][j]);
        }
    }
    return data;
}

matrice * soft_max_activation(matrice * data) {
    for (size_t i = 0; i < data->column_size(); i++)
    {
        double max = * max_element(begin(data->data[i]), end(data->data[i]));
        double sum = 0;
        for (size_t j = 0; j < data->row_size(); j++)
        {
            double exponated = exp(data->data[i][j] - max);
            sum += exponated;
            data->data[i][j] = exponated;
        }

        for (size_t j = 0; j < data->row_size(); j++)
        {
            data->data[i][j] = data->data[i][j] / sum;
        }
    }

    return data;
}