#ifndef __ACTIVATION
#define __ACTIVATION

#include "matrice_lib.h"
#include <functional>

class activaion_function
{
protected:
    Matrice * inputs;
    Matrice * derived_inputs;
    Matrice * outputs;
public:
    activaion_function();
    ~activaion_function();

    void forward(Matrice * inputs) {}
    void backward(Matrice * derived_inputs) {}

    Matrice * get_outputs() const;
    Matrice * get_derived_inputs() const;

    Matrice * apply_function(Matrice * data, double (*func)(double)) const;
};


class linear_activation : public activaion_function 
{
public:

    void forward(Matrice * inputs);
    void backward(Matrice * derived_inputs);
};

class sigmoid_activation : public activaion_function 
{
public:

    void forward(Matrice * inputs);
    void backward(Matrice * derived_inputs);
};

class ReLU_activation : public activaion_function 
{
public:

    void forward(Matrice * inputs);
    void backward(Matrice * derived_inputs);
};

class soft_max_activation : public activaion_function 
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