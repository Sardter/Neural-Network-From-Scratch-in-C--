#include "layer_dens.h"
#include "matrice_lib.h"
#include "layer_tools.h"
#include <vector>

using namespace std;

layer_dens::layer_dens(
        int input_num, int neuron_num, 
        double WR_L1, double WR_L2, 
        double BR_L1, double BR_L2
    ) : layer()
{
    this->weights = generate_gaugian_weights(input_num, neuron_num);
    this->biases = default_biases(neuron_num);

    this->biases_derivative = nullptr;

    this->BR_L1 = BR_L1; this->BR_L2 = BR_L2;
    this->WR_L1 = WR_L1; this->WR_L2 = WR_L2;
}

layer_dens::layer_dens(
        Matrice * weights, Vector * biases, 
        double WR_L1, double WR_L2, 
        double BR_L1, double BR_L2) : layer()
{
    this->weights = weights;
    this->biases = biases;

    this->biases_derivative = nullptr;

    this->BR_L1 = BR_L1; 
    this->BR_L2 = BR_L2;
    this->WR_L1 = WR_L1; 
    this->WR_L2 = WR_L2;
}

layer_dens::~layer_dens() {
    if (this->biases != nullptr) delete this->biases;
    if (this->biases_derivative != nullptr) delete this->biases_derivative;
}

void layer_dens::forward(Matrice * inputs) {
    this->output = inputs->dot_product(this->weights)->add(this->biases);
    this->inputs = inputs->copy();
}

void layer_dens::backward(Matrice * derivated_inputs) {
    this->weights_derivative = this->inputs->transpose()->dot_product(derivated_inputs);
    this->biases_derivative = derivated_inputs->columns_sum();

    if (this->WR_L1 > 0) {
        for (size_t i = 0; i < this->weights_derivative->column_size(); i++)
        {
            for (size_t j = 0; j < this->weights_derivative->row_size(); j++)
            {
                double value = this->WR_L1;
                if (this->weights->data[i][j] < 0) value *= -1;
                this->weights_derivative->data[i][j] += value;
            }
        }
    }

    if (this->BR_L1 > 0) {
        for (size_t i = 0; i < this->biases_derivative->size(); i++)
        {
            double value = this->WR_L2;
            if (this->biases->data[i] < 0) value *= -1;
            this->biases_derivative->data[i] += value;
        } 
    }

    if (this->WR_L2 > 0) {
       this->weights_derivative = this->weights_derivative->add(this->weights->product(this->WR_L2 * 2));
    }

    if (this->BR_L2 > 0) {
        this->biases_derivative = this->biases_derivative->add(this->biases->product(this->WR_L2 * 2));
    }

    this->inputs_derivative = derivated_inputs->dot_product(this->weights->transpose());
} 


Vector * layer_dens::get_biases_derivatives() const {
    return this->biases_derivative;
}

Vector * layer_dens::get_biases() const {
    return this->biases;
}

void layer_dens::set_biases(Vector * v) {
    this->biases = v;
}

double layer_dens::get_weight_regularizer_L1() const 
{
    return this->WR_L1;
}

double layer_dens::get_weight_regularizer_L2() const 
{
    return this->WR_L2;
}

double layer_dens::get_bias_regularizer_L1() const 
{
    return this->BR_L1;
}

double layer_dens::get_bias_regularizer_L2() const 
{
    return this->BR_L2;
}
