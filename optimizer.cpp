#include "layer_dens.h"
#include "matrice_lib.h"
#include "optimizer.h"
#include <stdexcept>
#include <vector>

using namespace std;

Optimizer::Optimizer(double learning_rate) {
    this->learning_rate = learning_rate;
}

Optimizer::~Optimizer() {

}

void Optimizer::pre_update() {

}

void Optimizer::post_update() {

}

double Optimizer::get_learning_rate() const {
    return this->learning_rate;
}

void Optimizer::update_params(Layer_Dens * layer) {
    throw invalid_argument("Abstract Function Called");
}

Stochastic_Gradient_Descent::Stochastic_Gradient_Descent(
    double learning_rate, 
    double decay,
    double momentum) : Optimizer(learning_rate) {
        this->current_learning_rate = learning_rate;
        this->decay = decay;
        this->iterations = 0;

        this->momentum = momentum;
        this->weight_momentum = nullptr;
        this->bias_momentum = nullptr;
}

Stochastic_Gradient_Descent::~Stochastic_Gradient_Descent() {
    if (this->weight_momentum != nullptr) delete this->weight_momentum;
    if (this->bias_momentum != nullptr) delete this->bias_momentum;
}

void Stochastic_Gradient_Descent::update_params(Layer_Dens * layer) {
    Matrice * weights = layer->get_weights();
    Matrice * weights_der = layer->get_weight_derivatives();

    Vector * biases = layer->get_biases();
    Vector * biases_der = layer->get_biases_derivatives();

    size_t weight_col = weights->column_size();
    size_t weight_row = weights->row_size();

    Matrice * weights_update = nullptr;
    Vector * biases_update = nullptr;

    if (this->momentum) {
        if (this->weight_momentum == nullptr) {
            this->weight_momentum = new Matrice(weight_col, weight_row);
            this->bias_momentum = new Vector(biases->size());
        }

        weights_update = this->weight_momentum->product(
            this->momentum)->subtract(weights_der->product(this->current_learning_rate));
        this->weight_momentum = weights_update;

        biases_update = this->bias_momentum
            ->product(this->momentum)
            ->subtract(biases_der
            ->product(this->current_learning_rate));
        this->bias_momentum = biases_update;
    } else {
        weights_update = weights_der->product(-1 * this->current_learning_rate);
        biases_update = biases_der->product(-1 * this->current_learning_rate);
    }

    layer->set_weights(weights->add(weights_update));
    layer->set_biases(biases->add(biases_update));
}

void Stochastic_Gradient_Descent::pre_update() {
    if (this->decay) {
        this->current_learning_rate = this->learning_rate * (1. / (1. + this->decay * this->iterations));
    }
}

void Stochastic_Gradient_Descent::post_update() {
    this->iterations++;
}

double Stochastic_Gradient_Descent::get_learning_rate() const {
    return this->current_learning_rate;
}