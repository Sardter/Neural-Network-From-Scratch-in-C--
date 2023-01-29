#include "matrice_lib.h"
#include <cmath>
#include "loss_functions.h"
#include <numeric>
#include "layer_dens.h"

using namespace std;

Loss_Function::Loss_Function() {
    this->inputs_derivatives = nullptr;
}

Loss_Function::~Loss_Function() {
    if (this->inputs_derivatives != nullptr) delete this->inputs_derivatives;
}

double Loss_Function::loss_percentage(Matrice * input, Vector * target) {
    Vector losses = *this->forward(input, target);
    return accumulate(losses.data.begin(), losses.data.end(), 0) / losses.size();
}

double Loss_Function::loss_percentage(Matrice * input, Matrice * target) {
    Vector losses = *this->forward(input, target);
    return accumulate(losses.data.begin(), losses.data.end(), 0) / losses.size();
}

Matrice * Loss_Function::get_input_derivatives() const {
    return this->inputs_derivatives;
}


double clip(double x) {
    if (x < 1e-7) return 1e-7;
    else if (x > 1 - 1e-7) return 1 - 1e-7;
    return x;
}

Vector * categorical_cross_entropy::forward(Matrice * input, Vector * target) {
    size_t col = input->column_size();
    
    Vector * res = new Vector(col);
    for (size_t i = 0; i < col; i++)
    {
        res->data[i] = -log(clip(input->data[i][target->data[i]]));
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

void categorical_cross_entropy::backward(Matrice * input, Vector * target) {
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
        res->data[i] = sum / col_size;
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

Vector * mean_squared_error::forward(Matrice * input, Matrice * target) {
    size_t col_size = input->column_size();
    size_t row_size = input->row_size();

    Vector * res = new Vector(col_size);

    for (size_t i = 0; i < col_size; i++)
    {
        double sum = 0;
        for (size_t j = 0; j < row_size; j++)
        {
            sum += (target->data[i][j] - input->data[i][j]) * (target->data[i][j] - input->data[i][j]);
        }
        res->data[i] = sum / col_size;
    }
    
    return res;
}

void mean_squared_error::backward(Matrice * input, Matrice * target) {    
    this->inputs_derivatives = input->subtract(target)->product(-2/input->row_size());
}