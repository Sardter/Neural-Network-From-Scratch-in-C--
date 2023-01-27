#include "matrice_lib.h"
#include <cmath>
#include "loss_functions.h"
#include "layer_dens.h"
#include <numeric>

using namespace std;

loss_function::loss_function() {
    this->inputs_derivatives = nullptr;
}

loss_function::~loss_function() {
    if (this->inputs_derivatives != nullptr) delete this->inputs_derivatives;
}

double loss_function::loss_percentage(Matrice * input, vector<int> * target) {
    Vector losses = *this->forward(input, target);
    return accumulate(losses.data.begin(), losses.data.end(), 0) / losses.size();
}

double loss_function::loss_percentage(Matrice * input, Matrice * target) {
    Vector losses = *this->forward(input, target);
    return accumulate(losses.data.begin(), losses.data.end(), 0) / losses.size();
}

double loss_function::regularization_loss(layer_dens * layer) const
{
    double loss = 0;

    if (layer->get_weight_regularizer_L1() > 0) {
        loss += layer->get_weight_regularizer_L1() * layer->get_weights()->abs()->sum();
    }

    if (layer->get_weight_regularizer_L2() > 0) {
        loss += layer->get_bias_regularizer_L2() * layer->get_weights()->product(layer->get_weights())->sum();
    }

    if (layer->get_bias_regularizer_L1() > 0) {
        loss += layer->get_bias_regularizer_L1() * layer->get_biases()->abs()->sum();
    }

    if (layer->get_bias_regularizer_L2() > 0) {
        loss += layer->get_bias_regularizer_L2() * layer->get_biases()->product(layer->get_biases())->sum();
    }

    return loss;
}

double clip(double x) {
    if (x < 1e-7) return 1e-7;
    else if (x > 1 - 1e-7) return 1 - 1e-7;
    return x;
}

Vector * categorical_cross_entropy::forward(Matrice * input, vector<int> * target) {
    size_t col = input->column_size();
    
    Vector * res = new Vector(col);
    for (size_t i = 0; i < col; i++)
    {
        res->data[i] = -log(clip(input->data[i][target->at(i)]));
    }
    return res;
}


Vector * categorical_cross_entropy::forward(Matrice * input, Matrice * target) {
    Vector * res = input->dot_product(target)->rows_sum();

    for (size_t i = 0; i < res->size(); i++)
    {
        res->data[i] = -log(clip(res->data[i]));
    }
    
    return res;
}

void categorical_cross_entropy::backward(Matrice * input, vector<int> * target) {
    this->backward(input, discrete_to_one_hot(target, input->row_size()));
}


void categorical_cross_entropy::backward(Matrice * input, Matrice * target) {
   this->inputs_derivatives = target->product(-1)->division(input)->division(input->column_size());
}


Vector * binary_cross_entropy::forward(Matrice * input, Matrice * target) {
    // target == [[1], [0], [1], [0]]
    size_t col_size = input->column_size();
    size_t row_size = input->row_size();

    Vector * res = new Vector(col_size);

    for (size_t i = 0; i < col_size; i++)
    {
        double sum = 0;
        for (size_t j = 0; j < row_size; j++)
        {
            sum += - (target->data[i][0] * log(clip(input->data[i][j]))) 
                + (1 - target->data[i][0]) * (1 - log(clip(input->data[i][j])));
        }
        res->data[i] = sum;
    }
    
    return res;
}

void binary_cross_entropy::backward(Matrice * input, Matrice * target) {
    // target == [[1], [0], [1], [0]]
    size_t col_size = input->column_size();
    size_t row_size = input->row_size();

    Matrice * res = input->copy();
    
    for (size_t i = 0; i < col_size; i++)
    {
        for (size_t j = 0; j < row_size; j++)
        {
            res->data[i][j] = (- (target->data[i][0] / clip(input->data[i][j]) - 
                (1 - target->data[i][0]) / (1 - clip(input->data[i][j]))) / row_size) / col_size;
        }
        
    }
    
    this->inputs_derivatives = res;
}