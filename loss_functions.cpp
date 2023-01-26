#include "matrice_lib.h"
#include <cmath>
#include "loss_functions.h"
#include <numeric>

using namespace std;

loss_function::loss_function() {
    this->inputs_derivatives = nullptr;
}

loss_function::~loss_function() {
    if (this->inputs_derivatives != nullptr) delete this->inputs_derivatives;
}

double loss_function::loss_percentage(matrice * input, vector<int> * target) {
    vector<double> losses = *this->forward(input, target);
    return accumulate(losses.begin(), losses.end(), 0) / losses.size();
}

double loss_function::loss_percentage(matrice * input, matrice * target) {
    vector<double> losses = *this->forward(input, target);
    return accumulate(losses.begin(), losses.end(), 0) / losses.size();
}

double clip(double x) {
    if (x < 1e-7) return 1e-7;
    else if (x > 1 - 1e-7) return 1 - 1e-7;
    return x;
}

vector<double> * categorical_cross_entropy::forward(matrice * input, vector<int> * target) {
    size_t col = input->column_size();
    
    vector<double> * res = new vector<double>(col, 0);
    for (size_t i = 0; i < col; i++)
    {
        res->at(i) = -log(clip(input->data[i][target->at(i)]));
    }
    return res;
}


vector<double> * categorical_cross_entropy::forward(matrice * input, matrice * target) {
    vector<double> * res = input->product(target)->rows_sum();

    for (size_t i = 0; i < res->size(); i++)
    {
        res->at(i) = -log(clip(res->at(i)));
    }
    
    return res;
}

void categorical_cross_entropy::backward(matrice * input, vector<int> * target) {
    this->backward(input, discrete_to_one_hot(target, input->row_size()));
}


void categorical_cross_entropy::backward(matrice * input, matrice * target) {
   this->inputs_derivatives = target->product(-1)->division(input)->division(input->column_size());
}

