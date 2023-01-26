#include "layer_dens.h"
#include "matrice_lib.h"
#include "layer_tools.h"
#include <vector>

using namespace std;

layer_dens::layer_dens(int input_num, int neuron_num) {
    this->weights = generate_rondom_weights(input_num, neuron_num);
    this->biases = default_biases(neuron_num);
    this->inputs = nullptr;

    this->output = nullptr;

    this->biases_derivative = nullptr;
    this->inputs_derivative = nullptr;
    this->weights_derivative = nullptr;
}

layer_dens::layer_dens(matrice * weights, vector<double> * biases) {
    this->weights = weights;
    this->biases = biases;
    this->inputs = nullptr;

    this->output = nullptr;

    this->biases_derivative = nullptr;
    this->inputs_derivative = nullptr;
    this->weights_derivative = nullptr;
}

layer_dens::~layer_dens() {
    if (this->weights != nullptr) delete this->weights;
    if (this->biases != nullptr) delete this->biases;
    if (this->output != nullptr) delete this->output;
    if (this->inputs != nullptr) delete this->inputs;
    if (this->biases_derivative != nullptr) delete this->biases_derivative;
    if (this->inputs_derivative != nullptr) delete this->inputs_derivative;
    if (this->weights_derivative != nullptr) delete this->weights_derivative;
}

void layer_dens::forward(matrice * inputs) {
    this->output = inputs->product(this->weights)->add(this->biases);
    this->inputs = inputs->copy();
}

void layer_dens::backward(matrice * derivated_inputs) {
    this->weights_derivative = this->inputs->transpose()->product(derivated_inputs);
    this->inputs_derivative = derivated_inputs->product(this->weights->transpose());

    this->biases_derivative = derivated_inputs->columns_sum();
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

matrice * layer_dens::get_weight_derivatives() const {
    return this->weights_derivative;
}
matrice * layer_dens::get_inputs_derivatives() const {
    return this->inputs_derivative;
}

matrice * layer_dens::get_inputs() const {
    return this->inputs;
}

vector<double> * layer_dens::get_biases_derivatives() const {
    return new vector<double>(*this->biases_derivative);
}