#ifndef __ACTIVATION
#define __ACTIVATION

#include "matrice_lib.h"
#include <functional>

class activaion_function
{
protected:
    matrice * inputs;
    matrice * derived_inputs;
    matrice * outputs;
public:
    activaion_function();
    ~activaion_function();

    void forward(matrice * inputs) {}
    void backward(matrice * derived_inputs) {}

    matrice * get_outputs() const;
    matrice * get_derived_inputs() const;

    matrice * apply_function(matrice * data, double (*func)(double)) const;
};


class step_activation : public activaion_function 
{
public:

    void forward(matrice * inputs);
    void backward(matrice * derived_inputs);
};

class sigmoid_activation : public activaion_function 
{
public:

    void forward(matrice * inputs);
    void backward(matrice * derived_inputs);
};

class ReLU_activation : public activaion_function 
{
public:

    void forward(matrice * inputs);
    void backward(matrice * derived_inputs);
};

class soft_max_activation : public activaion_function 
{
public:

    void forward(matrice * inputs);
    void backward(matrice * derived_inputs);
};

double step_function(double x);

double step_function_derivative(double x);

double sigmoid_function(double x);

double sigmoid_function_derivative(double x);

double ReLU_function(double x);

double ReLU_function_derivative(double x);

#endif