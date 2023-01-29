#include "layer_dropout.h"
#include "layer_tools.h"
#include "matrice_lib.h"
#include "activaion_functions.h"

Layer_Dropout::Layer_Dropout(double rate, Activaion_Function * function) : Layer(function) 
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

void Layer_Dropout::forward(Matrice * inputs, bool is_trainig) 
{
    this->inputs = inputs->copy();
    if (is_trainig) {
        this->binary_mask = generate_binomial_weights(
            inputs->column_size(), inputs->row_size(), this->rate)->division(this->rate);
        this->output = this->binary_mask->product(inputs);
    } else {
        this->output = inputs->copy();
    }
}

void Layer_Dropout::backward(Matrice * derived_inputs) {
    this->inputs_derivative = derived_inputs->product(this->inputs);
}
