#include "layer_dens.h"
#include "matrice_lib.h"
#include "layer_tools.h"
#include <vector>

using namespace std;

layer_dens::layer_dens(int input_num, int neuron_num) {
    this->weights = generate_rondom_weights(input_num, neuron_num);
    this->biases = default_biases(neuron_num);
}

layer_dens::~layer_dens() {
    delete this->weights;
    delete this->biases;
    if (this->output != nullptr) delete this->output;
}

void layer_dens::forward(matrice * inputs) {
    this->output = vector_addition(
        matrice_multiplication(inputs, this->weights), this->biases);
}

matrice * layer_dens::get_output() const {
    return this->output;
}

matrice * layer_dens::get_weights() const {
    return this->weights;
}

vector<double> * layer_dens::get_biases() const {
    return this->biases;
}