#include "layer.h"
#include "matrice_lib.h"
#include <stdexcept>

Layer::Layer() {
    this->weights = nullptr;
    this->inputs = nullptr;

    this->output = nullptr;

    this->inputs_derivative = nullptr;
    this->weights_derivative = nullptr;
}

Layer::~Layer() {
    if (this->weights != nullptr) delete this->weights;
    if (this->output != nullptr) delete this->output;
    if (this->inputs != nullptr) delete this->inputs;
    if (this->inputs_derivative != nullptr) delete this->inputs_derivative;
    if (this->weights_derivative != nullptr) delete this->weights_derivative;
}

Matrice * Layer::get_output() const {
    return this->output;
}

Matrice * Layer::get_weights() const {
    return this->weights;
}

Matrice * Layer::get_weight_derivatives() const {
    return this->weights_derivative;
}
Matrice * Layer::get_inputs_derivatives() const {
    return this->inputs_derivative;
}

Matrice * Layer::get_inputs() const {
    return this->inputs;
}

void Layer::set_weights(Matrice * m) {
    this->weights = m;
}

void Layer::forward(Matrice * inputs) {
    throw invalid_argument("Unimplemented");
}

void Layer::backward(Matrice * derivated_inputs) {
    throw invalid_argument("Unimplemented");
}