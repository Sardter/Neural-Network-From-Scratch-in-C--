#include "activaion_functions.h"
#include <cstdlib>
#include <functional>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

Activaion_Function::Activaion_Function() 
{
    this->inputs = nullptr;
    this->outputs = nullptr;
    this->derived_inputs = nullptr;
}

Activaion_Function::~Activaion_Function() 
{
    if (this->inputs != nullptr) delete this->inputs;
    if (this->outputs != nullptr) delete this->outputs;
    if (this->derived_inputs != nullptr) delete this->derived_inputs;
}

Matrice * Activaion_Function::apply_function(Matrice * data, double (*func)(double)) const
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

Matrice * Activaion_Function::get_derived_inputs() const
{
    return this->derived_inputs;
}

Matrice * Activaion_Function::get_outputs() const
{
    return this->outputs;
}

void Linear_Activation::forward(Matrice * inputs)
{
    this->inputs = inputs->copy();
    this->outputs = inputs->copy();
}

void Linear_Activation::backward(Matrice * derived_inputs) 
{
   this->derived_inputs = derived_inputs->copy();
}

Matrice * Linear_Activation::predictions(Matrice * outputs) const
{
    return outputs->copy();
}

void Sigmoid_Activation::forward(Matrice * inputs) 
{
    this->inputs = inputs->copy();
    this->outputs = this->apply_function(inputs , sigmoid_function);
}

void Sigmoid_Activation::backward(Matrice * derived_inputs) 
{
    this->derived_inputs = derived_inputs->product(this->outputs->product(-1)->add(1)->product(this->outputs));
}

Matrice * Sigmoid_Activation::predictions(Matrice * outputs) const
{
    Matrice * res = outputs->copy();

    size_t col_size = derived_inputs->column_size();
    size_t row_size = derived_inputs->row_size();

    for (size_t i = 0; i < col_size; i++)
    {
        for (size_t j = 0; j < row_size; j++)
        {
            if (res->data[i][j] < 0) 
                res->data[i][j] = 0;
        }
        
    }
    
    return res;
}

void ReLU_Activation::forward(Matrice * inputs) 
{
    this->inputs = inputs->copy();
    this->outputs = this->apply_function(inputs , ReLU_function);
}

void ReLU_Activation::backward(Matrice * derived_inputs) 
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

Matrice * ReLU_Activation::predictions(Matrice * outputs) const
{
    return outputs->copy();
}

void Softmax_Activation::forward(Matrice * inputs)
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

void Softmax_Activation::backward(Matrice * derived_inputs)
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

Matrice * Softmax_Activation::predictions(Matrice * outputs) const
{
    size_t col_size = outputs->column_size();
    size_t row_size = outputs->row_size();

    Matrice * res = new Matrice(col_size, 1);
    for (size_t i = 0; i < col_size; i++)
    {
        res->data[i][0] = max_element(outputs->data[i].begin(), outputs->data[i].end()) - outputs->data[i].begin();
    }
    
    return res;
}

double sigmoid_function(double x) {
    return 1 / (1 + exp(-x));
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
