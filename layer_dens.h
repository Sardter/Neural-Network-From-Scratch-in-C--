#ifndef __LAYER_DENSE
#define __LAYER_DENSE

#include "matrice_lib.h"
#include "vector"

class layer_dens
{
private:
    matrice * weights;
    matrice * inputs;
    vector<double> * biases;

    matrice * output;

    matrice * weights_derivative;
    matrice * inputs_derivative;
    vector<double> * biases_derivative;
public:
    layer_dens(int input_num, int neoron_num);
    layer_dens(matrice * weights, vector<double> * biases);
    ~layer_dens();

    void forward(matrice * inputs);
    void backward(matrice * derivated_inputs);

    matrice * get_weights() const;
    vector<double> * get_biases() const;
    matrice * get_output() const;
    matrice * get_weight_derivatives() const;
    matrice * get_inputs_derivatives() const;
    matrice * get_inputs() const;
    vector<double> * get_biases_derivatives() const;
};


#endif