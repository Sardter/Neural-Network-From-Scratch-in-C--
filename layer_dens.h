#ifndef __LAYER_DENSE
#define __LAYER_DENSE

#include "layer.h"
#include "matrice_lib.h"
#include "vector"

class layer_dens: public layer
{
private:
    Vector * biases;

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

    Vector * get_biases() const;
    Vector * get_biases_derivatives() const;

    double get_weight_regularizer_L1() const;
    double get_weight_regularizer_L2() const;
    double get_bias_regularizer_L1() const;
    double get_bias_regularizer_L2() const;

    void set_biases(Vector * v);
};


#endif