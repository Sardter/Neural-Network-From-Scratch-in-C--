#ifndef __LAYER_TOOLS
#define __LAYER_TOOLS

#include "matrice_lib.h"
#include "activaion_functions.h"
#include "loss_functions.h"

using namespace std;

class Activation_Softmax_Loss_Catogorical_Crossentropy {
private:
    Matrice * output;
    Matrice * derived_inputs;
    Softmax_Activation activation;
    categorical_cross_entropy loss;
public:
    Activation_Softmax_Loss_Catogorical_Crossentropy();
    ~Activation_Softmax_Loss_Catogorical_Crossentropy();

    double forward(Matrice * inputs, vector<int> * targets);
    void backward(Matrice * inputs, vector<int> * targets);

    Matrice * get_output() const;
    Matrice * get_derived_inputs() const;
};


Matrice * generate_gaugian_weights(int columns, int row, double rate = 0.1);

Matrice * generate_binomial_weights(int columns, int row, double rate);

Vector * default_biases(int neurons);

double calculate_clasification_accuracy(Matrice * data, vector<int> * targets);

double calculate_regression_accuracy(Matrice * data, Matrice * targets);

double accuracy_percision(Matrice * data, double divisor = 1);

#endif