#ifndef __LAYER_TOOLS
#define __LAYER_TOOLS

#include "matrice_lib.h"
#include "activaion_functions.h"
#include "loss_functions.h"

using namespace std;

class activation_softmax_loss_catogorical_crossentropy {
private:
    matrice * output;
    matrice * derived_inputs;
    soft_max_activation activation;
    categorical_cross_entropy loss;
public:
    activation_softmax_loss_catogorical_crossentropy();
    ~activation_softmax_loss_catogorical_crossentropy();

    double forward(matrice * inputs, vector<int> * targets);
    void backward(matrice * inputs, vector<int> * targets);

    matrice * get_output() const;
    matrice * get_derived_inputs() const;
};


matrice * generate_rondom_weights(int rows, int columns);

vector<double> * default_biases(int neurons);

double calculate_accuracy(matrice * data, vector<int> * targets);

#endif