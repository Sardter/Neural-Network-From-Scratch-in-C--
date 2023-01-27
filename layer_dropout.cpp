#include "layer_dropout.h"
#include "layer_tools.h"
#include "matrice_lib.h"

layer_dropout::layer_dropout(double rate) : layer() 
{
    this->rate = 1 - rate;
}

double layer_dropout::get_rate() const
{
    return this->rate;
}

void layer_dropout::set_rate(double rate)
{
    this->rate = rate;
}

void layer_dropout::forward(Matrice * inputs) 
{
    this->inputs = inputs->copy();
    this->weights = generate_binomial_weights(
        inputs->column_size(), inputs->row_size(), this->rate)->division(this->rate);
    this->output = this->weights->product(inputs);
}

void layer_dropout::backward(Matrice * derived_inputs) {
    this->inputs_derivative = derived_inputs->product(this->inputs);
}