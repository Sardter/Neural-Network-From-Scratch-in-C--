#include "layer.h"
#include "matrice_lib.h"
#include "activaion_functions.h"
#include <stdexcept>

Layer::Layer(Activaion_Function * function) {   
    this->inputs = nullptr;
    this->output = nullptr;  
    this->activation_function = function;
}

Layer::~Layer() {
    if (this->output != nullptr) delete this->output;
    if (this->inputs != nullptr) delete this->inputs;
    if (this->activation_function != nullptr) delete this->activation_function;
}

Matrice * Layer::get_output() const {
    return this->output;
}

Matrice * Layer::get_inputs() const {
    return this->inputs;
}

Matrice * Layer::get_inputs_derivatives() const {
    return this->inputs_derivative;
}

Activaion_Function * Layer::get_activation() const {
    return this->activation_function;
}

void Layer::backward(Matrice * derivated_inputs) {
    throw invalid_argument("Unimplemented");
}

void Layer::forward(Matrice * inputs, bool is_training) {
    this->inputs = inputs;
    this->output = inputs;
}

bool Layer::is_trainable() const {
    return false;
}

double Layer::get_regularization_loss() const {
    return 0;
}

ostream& operator << (ostream& stream, Layer l) {
    stream << "------ Layer ------" << endl;
    stream << "Inputs: " << endl;
    stream << *l.get_inputs() << endl;
    stream << "Outputs: " << endl;
    stream << *l.get_output() << endl;
    stream << "-------------------" << endl;
    return stream;
}