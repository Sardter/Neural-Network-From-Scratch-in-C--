#include "activaion_functions.h"
#include <cstdlib>
#include <functional>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

activaion_function::activaion_function() 
{
    this->inputs = nullptr;
    this->outputs = nullptr;
    this->derived_inputs = nullptr;
}

activaion_function::~activaion_function() 
{
    if (this->inputs != nullptr) delete this->inputs;
    if (this->outputs != nullptr) delete this->outputs;
    if (this->derived_inputs != nullptr) delete this->derived_inputs;
}

Matrice * activaion_function::apply_function(Matrice * data, double (*func)(double)) const
{
    Matrice * res = data->copy();
    for (size_t i = 0; i < data->column_size(); i++)
    {
        for (size_t j = 0; j < data->row_size(); j++)
        {
            res->data[i][j] = func(data->data[i][j]);
        }
    }
    return res;
}

Matrice * activaion_function::get_derived_inputs() const
{
    return this->derived_inputs;
}

Matrice * activaion_function::get_outputs() const
{
    return this->outputs;
}

void step_activation::forward(Matrice * inputs)
{
    this->inputs = inputs->copy();
    this->outputs = this->apply_function(inputs , step_function);
}

void step_activation::backward(Matrice * derived_inputs) 
{
   // this->derived_inputs = this->apply_function(inputs, step_function_derivative);
}

void sigmoid_activation::forward(Matrice * inputs) 
{
    this->inputs = inputs->copy();
    this->outputs = this->apply_function(inputs , sigmoid_function);
}

void sigmoid_activation::backward(Matrice * derived_inputs) 
{
    //this->derived_inputs = this->apply_function(inputs, sigmoid_function_derivative);
}

void ReLU_activation::forward(Matrice * inputs) 
{
    this->inputs = inputs->copy();
    this->outputs = this->apply_function(inputs , ReLU_function);
}

void ReLU_activation::backward(Matrice * derived_inputs) 
{
    this->derived_inputs = derived_inputs->copy();

    size_t col_size = derived_inputs->column_size();
    size_t row_size = derived_inputs->row_size();

    for (size_t i = 0; i < col_size; i++)
    {
        for (size_t j = 0; j < row_size; j++)
        {
            if (this->inputs->data[i][j] < 0) 
                this->derived_inputs->data[i][j] = 0;
        }
    }
}

void soft_max_activation::forward(Matrice * inputs)
{
    this->inputs = inputs->copy();
    this->outputs = inputs->copy();

    size_t col_size = this->outputs->column_size();
    size_t row_size = this->outputs->row_size();

    for (size_t i = 0; i < col_size; i++)
    {
        vector<double> col = this->outputs->data[i];
        double max = * max_element(begin(col), end(col));
        double sum = 0;
        for (size_t j = 0; j < row_size; j++)
        {
            double exponated = exp(this->outputs->data[i][j] - max);
            sum += exponated;
            this->outputs->data[i][j] = exponated;
        }

        for (size_t j = 0; j < row_size; j++)
        {
            this->outputs->data[i][j] = this->outputs->data[i][j] / sum;
        }
    }
}

void soft_max_activation::backward(Matrice * derived_inputs)
{
    size_t col_size = derived_inputs->column_size();
    size_t row_size = derived_inputs->row_size();

    this->derived_inputs = new Matrice(col_size, row_size);
    for (size_t i = 0; i < col_size; i++)
    {
        Matrice * jacob = jacobian_matrice(&this->outputs->data[i])->
            dot_product((new Matrice(derived_inputs->data[i]))->transpose());
        
        this->derived_inputs->data[i] = jacob->get_column(0)->data;
    }
}

double step_function(double x) {
    return x > 0;
}

double step_function_derivative(double x) {
    return 0;
}

double sigmoid_function(double x) {
    return x / (1 + abs(x));
}

double sigmoid_function_derivative(double x) {
    double e = exp(x);

    return e / ((1 + e) * (1 + e));
}

double ReLU_function(double x) {
    return x > 0 ? x : 0;
}

double ReLU_function_derivative(double x) {
    return x > 0;
}
