#ifndef __LAYER
#define __LAYER

#include "matrice_lib.h"

class Layer
{
protected:
    Matrice * weights;
    Matrice * inputs;

    Matrice * output;

    Matrice * weights_derivative;
    Matrice * inputs_derivative;
public:
    Layer();
    ~Layer();

    void set_weights(Matrice * m);

    Matrice * get_inputs() const;
    Matrice * get_weights() const;
    Matrice * get_output() const;
    Matrice * get_weight_derivatives() const;
    Matrice * get_inputs_derivatives() const;

    void forward(Matrice * inputs);
    void backward(Matrice * derivated_inputs);
};


#endif