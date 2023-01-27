#include "layer_dropout.h"
#include "layer_tools.h"
#include "matrice_lib.h"

Layer_Dropout::Layer_Dropout(double rate) : Layer() 
{
    this->rate = 1 - rate;
}

double Layer_Dropout::get_rate() const
{
    return this->rate;
}

void Layer_Dropout::set_rate(double rate)
{
    this->rate = rate;
}

void Layer_Dropout::forward(Matrice * inputs) 
{
    this->inputs = inputs->copy();
    this->weights = generate_binomial_weights(
        inputs->column_size(), inputs->row_size(), this->rate)->division(this->rate);
    this->output = this->weights->product(inputs);
}

void Layer_Dropout::backward(Matrice * derived_inputs) {
    this->inputs_derivative = derived_inputs->product(this->inputs);
}