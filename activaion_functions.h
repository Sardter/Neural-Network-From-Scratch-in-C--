#ifndef __ACTIVATION
#define __ACTIVATION

#include "matrice_lib.h"
#include <functional>

class Activaion_Function
{
protected:
    Matrice * inputs;
    Matrice * derived_inputs;
    Matrice * outputs;
public:
    Activaion_Function();
    ~Activaion_Function();

    void forward(Matrice * inputs) {}
    void backward(Matrice * derived_inputs) {}

    Matrice * get_outputs() const;
    Matrice * get_derived_inputs() const;

    Matrice * apply_function(Matrice * data, double (*func)(double)) const;
};


class Linear_Activation : public Activaion_Function 
{
public:

    void forward(Matrice * inputs);
    void backward(Matrice * derived_inputs);
};

class Sigmoid_Activation : public Activaion_Function 
{
public:

    void forward(Matrice * inputs);
    void backward(Matrice * derived_inputs);
};

class ReLU_activation : public Activaion_Function 
{
public:

    void forward(Matrice * inputs);
    void backward(Matrice * derived_inputs);
};

class Softmax_Activation : public Activaion_Function 
{
public:

    void forward(Matrice * inputs);
    void backward(Matrice * derived_inputs);
};

double sigmoid_function(double x);

double sigmoid_function_derivative(double x);

double ReLU_function(double x);

double ReLU_function_derivative(double x);

#endif