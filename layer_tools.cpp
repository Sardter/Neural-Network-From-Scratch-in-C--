#include "layer_tools.h"
#include "matrice_lib.h"
#include "loss_functions.h"
#include "activaion_functions.h"
#include "random"
#include <stdexcept>

#include <algorithm>

using namespace std;

Activation_Softmax_Loss_Catogorical_Crossentropy::Activation_Softmax_Loss_Catogorical_Crossentropy() {
    this->output = nullptr;
    this->derived_inputs = nullptr;
}

Activation_Softmax_Loss_Catogorical_Crossentropy::~Activation_Softmax_Loss_Catogorical_Crossentropy() {
    if (this->output != nullptr) delete this->output;
    if (this->derived_inputs != nullptr) delete this->derived_inputs;
}

Matrice * Activation_Softmax_Loss_Catogorical_Crossentropy::get_output() const {
    return this->output;
}

Matrice * Activation_Softmax_Loss_Catogorical_Crossentropy::get_derived_inputs() const {
    return this->derived_inputs;
}

double Activation_Softmax_Loss_Catogorical_Crossentropy::forward(Matrice * inputs, Vector * targets) {
    this->activation.forward(inputs);
    this->output = this->activation.get_outputs();

    return this->loss.loss_percentage(inputs, targets);
}

void Activation_Softmax_Loss_Catogorical_Crossentropy::backward(Matrice * inputs, Vector * targets) {
    cout << "here" << endl;
    this->derived_inputs = inputs->copy();
    for (size_t i = 0; i < inputs->column_size(); i++)
    {
        this->derived_inputs->data[i][targets->data[i]] -= 1;
    }

    this->derived_inputs = this->derived_inputs->division(inputs->column_size());
}

Matrice * generate_gaugian_weights(int columns, int rows, double rate) {
    Matrice * weights = new Matrice(columns, rows);

    default_random_engine generator(0);
    normal_distribution<double> dist(0, 1);

    for (size_t i = 0; i < columns; i++)
    {
        for (size_t j = 0; j < rows; j++)
        {
            weights->data[i][j] = rate * dist(generator);
        }
    }
    return weights;
}

Matrice * generate_binomial_weights(int columns, int rows, double rate) {
    Matrice * weights = new Matrice(columns, rows);

    default_random_engine generator(0);
    binomial_distribution<double> dist(1, rate);

    for (size_t i = 0; i < columns; i++)
    {
        for (size_t j = 0; j < rows; j++)
        {
            weights->data[i][j] = 0.1 * dist(generator);
        }
    }

    return weights;
}

Vector * default_biases(int neurons) {
    return new Vector(neurons);
}

Accuracy::Accuracy() {

}

Accuracy::~Accuracy() {

}

bool Accuracy::compare(Matrice * data, Matrice * target, int i, int j) {
    throw invalid_argument("Not implemented");
}

bool Accuracy::compare(Matrice * data, Vector * target, int i) {
    throw invalid_argument("Not implemented");
}

void Accuracy::prepare(Matrice * targets, bool reprepare) {

}

void Accuracy::prepare(Vector * targets, bool reprepare) {

}

double Accuracy::calculate(Matrice * data, Matrice * targets) {
    size_t col_size = data->column_size();
    size_t row_size = data->row_size();

    double sum = 0;
    for (size_t i = 0; i < col_size; i++)
    {
        for (size_t j = 0; j < row_size; j++)
        {
            sum += this->compare(data, targets, i, j);
        }
    }
    
    return sum / (col_size * row_size);
}

double Accuracy::calculate(Matrice * data, Vector * targets) {
    double sum = 0;
    for (size_t i = 0; i < data->column_size(); i++)
    {
        sum += this->compare(data, targets, i);
    }
    
    return sum / data->column_size();
}

Accuracy_Regression::Accuracy_Regression() {
    this->percision = 0;
}

void Accuracy_Regression::prepare(Matrice * targets, bool reprepare) {
    if (!this->percision || reprepare) {
        this->percision = accuracy_percision(targets, 250);
    }
}

bool Accuracy_Regression::compare(Matrice * data, Matrice * target, int i, int j) {
    double acc = data->data[i][j] - target->data[i][0];
    return acc < this->percision && acc > -this->percision;
}

bool Accuracy_Classification::compare(Matrice * data, Vector * targets, int i) {
    return targets->data[i] == max_element(data->data[i].begin(), data->data[i].end()) - data->data[i].begin();
}

bool Accuracy_Classification::compare(Matrice * data, Matrice * targets, int i, int j) {
    double target_max = max_element(targets->data[i].begin(), targets->data[i].end()) - targets->data[i].begin();
    double data_max = max_element(data->data[i].begin(), data->data[i].end()) - data->data[i].begin();
    return data_max == target_max;
}

double Accuracy_Classification::calculate(Matrice * data, Matrice * targets) {
    double sum = 0;
    for (size_t i = 0; i < data->column_size(); i++)
    {
        sum += this->compare(data, targets, i, 0);
    }
    
    return sum / data->column_size();
}

double calculate_clasification_accuracy(Matrice * data, Vector * targets) {
    double sum = 0;
    for (size_t i = 0; i < data->column_size(); i++)
    {
        sum += targets->data[i] == max_element(data->data[i].begin(), data->data[i].end()) - data->data[i].begin();
    }
    
    return sum / data->column_size();
}

double accuracy_percision(Matrice * data, double divisor) {
    return standard_deviation(data) / divisor;
}

double calculate_regression_accuracy(Matrice * data, Matrice * targets) {
    size_t col_size = data->column_size();
    size_t row_size = data->row_size();

    double sum = 0;
    double percision = accuracy_percision(data, 250);
    for (size_t i = 0; i < col_size; i++)
    {
        for (size_t j = 0; j < row_size; j++)
        {
            double acc = data->data[i][j] - targets->data[i][0];
            sum += acc < percision && acc > -percision;
        }
    }
    
    return sum / (col_size * row_size);
}