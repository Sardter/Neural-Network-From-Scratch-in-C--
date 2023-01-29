#ifndef __LAYER
#define __LAYER

#include "matrice_lib.h"
#include "nueral_item.h"
#include "activaion_functions.h"

class Layer
{
protected: 
    Matrice * inputs;
    Matrice * output;
    Matrice * inputs_derivative;
    Activaion_Function * activation_function;
public:
    Layer(Activaion_Function * function);
    ~Layer();

    Matrice * get_inputs() const;
    Matrice * get_output() const;
    Matrice * get_inputs_derivatives() const;

    virtual void forward(Matrice * inputs, bool is_training);
    virtual void backward(Matrice * derivated_inputs);

    virtual bool is_trainable() const;
    Activaion_Function * get_activation() const;

    virtual double get_regularization_loss() const;
};

ostream& operator << (ostream& stream, Layer m);

#endif