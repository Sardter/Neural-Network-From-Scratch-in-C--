#include "layer.h"
#include "matrice_lib.h"
#include <stdexcept>

layer::layer() {
    this->weights = nullptr;
    this->inputs = nullptr;

    this->output = nullptr;

    this->inputs_derivative = nullptr;
    this->weights_derivative = nullptr;
}

layer::~layer() {
    if (this->weights != nullptr) delete this->weights;
    if (this->output != nullptr) delete this->output;
    if (this->inputs != nullptr) delete this->inputs;
    if (this->inputs_derivative != nullptr) delete this->inputs_derivative;
    if (this->weights_derivative != nullptr) delete this->weights_derivative;
}

Matrice * layer::get_output() const {
    return this->output;
}

Matrice * layer::get_weights() const {
    return this->weights;
}

Matrice * layer::get_weight_derivatives() const {
    return this->weights_derivative;
}
Matrice * layer::get_inputs_derivatives() const {
    return this->inputs_derivative;
}

Matrice * layer::get_inputs() const {
    return this->inputs;
}

void layer::set_weights(Matrice * m) {
    this->weights = m;
}

void layer::forward(Matrice * inputs) {
    throw invalid_argument("Unimplemented");
}

void layer::backward(Matrice * derivated_inputs) {
    throw invalid_argument("Unimplemented");
}