#include "layer_tools.h"
#include "matrice_lib.h"
#include "random"

#include <algorithm>

using namespace std;

matrice * generate_rondom_weights(int rows, int columns) {
    matrice * weights = new matrice(rows, columns);

    default_random_engine generator(0);
    normal_distribution<double> dist(0, 1);

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < columns; j++)
        {
            weights->data[i][j] = 0.1 * dist(generator);
        }
    }
    //cout << *weights;
    return weights;
}

vector<double> * default_biases(int neurons) {
    return new vector<double>(neurons, 0);
}

double calculate_accuracy(matrice * data, vector<int> * targets) {
    double sum = 0;
    for (size_t i = 0; i < data->column_size(); i++)
    {
        sum += targets->at(i) == max_element(begin(data->data[i]), end(data->data[i])) - begin(data->data[i]);
    }
    
    return sum / data->column_size();
}