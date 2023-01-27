#include "layer_tools.h"
#include "matrice_lib.h"
#include "loss_functions.h"
#include "activaion_functions.h"
#include "random"

#include <algorithm>

using namespace std;

activation_softmax_loss_catogorical_crossentropy::activation_softmax_loss_catogorical_crossentropy() {
    this->output = nullptr;
    this->derived_inputs = nullptr;
}

activation_softmax_loss_catogorical_crossentropy::~activation_softmax_loss_catogorical_crossentropy() {
    if (this->output != nullptr) delete this->output;
    if (this->derived_inputs != nullptr) delete this->derived_inputs;
}

Matrice * activation_softmax_loss_catogorical_crossentropy::get_output() const {
    return this->output;
}

Matrice * activation_softmax_loss_catogorical_crossentropy::get_derived_inputs() const {
    return this->derived_inputs;
}

double activation_softmax_loss_catogorical_crossentropy::forward(Matrice * inputs, vector<int> * targets) {
    this->activation.forward(inputs);
    this->output = this->activation.get_outputs();

    return this->loss.loss_percentage(inputs, targets);
}

void activation_softmax_loss_catogorical_crossentropy::backward(Matrice * inputs, vector<int> * targets) {
    cout << "here" << endl;
    this->derived_inputs = inputs->copy();
    for (size_t i = 0; i < inputs->column_size(); i++)
    {
        this->derived_inputs->data[i][targets->at(i)] -= 1;
    }

    this->derived_inputs = this->derived_inputs->division(inputs->column_size());
}

Matrice * generate_rondom_weights(int rows, int columns) {
    Matrice * weights = new Matrice(rows, columns);

    default_random_engine generator(0);
    normal_distribution<double> dist(0, 1);

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < columns; j++)
        {
            weights->data[i][j] = 0.1 * dist(generator);
        }
    }
    //cout << *weights;
    return weights;
}

Vector * default_biases(int neurons) {
    return new Vector(neurons);
}

double calculate_accuracy(Matrice * data, vector<int> * targets) {
    double sum = 0;
    for (size_t i = 0; i < data->column_size(); i++)
    {
        sum += targets->at(i) == max_element(begin(data->data[i]), end(data->data[i])) - begin(data->data[i]);
    }
    
    return sum / data->column_size();
}