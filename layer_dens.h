#ifndef __LAYER_DENSE
#define __LAYER_DENSE

#include "matrice_lib.h"
#include "vector"

class layer_dens
{
private:
    Matrice * weights;
    Matrice * inputs;
    Vector * biases;

    Matrice * output;

    Matrice * weights_derivative;
    Matrice * inputs_derivative;
    Vector * biases_derivative;

    // weight regularizer
    double WR_L1, WR_L2;
    // bias regularizer
    double BR_L1, BR_L2;
public:
    layer_dens(
        int input_num, int neoron_num, 
        double WR_L1 = 0, double WR_L2 = 0, 
        double BR_L1 = 0, double BR_L2 = 0
    );
    layer_dens(Matrice * weights, Vector * biases, 
        double WR_L1 = 0, double WR_L2 = 0, 
        double BR_L1 = 0, double BR_L2 = 0
    );
    ~layer_dens();

    void forward(Matrice * inputs);
    void backward(Matrice * derivated_inputs);

    Matrice * get_weights() const;
    Vector * get_biases() const;
    Matrice * get_output() const;
    Matrice * get_weight_derivatives() const;
    Matrice * get_inputs_derivatives() const;
    Matrice * get_inputs() const;
    Vector * get_biases_derivatives() const;

    double get_weight_regularizer_L1() const;
    double get_weight_regularizer_L2() const;
    double get_bias_regularizer_L1() const;
    double get_bias_regularizer_L2() const;

    void set_weights(Matrice * m);
    void set_biases(Vector * v);
};


#endif