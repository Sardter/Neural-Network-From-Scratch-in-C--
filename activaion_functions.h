#ifndef __ACTIVATION
#define __ACTIVATION

#include "matrice_lib.h"
#include <functional>
#include <stdexcept>

class Activaion_Function
{
protected:
    Matrice * inputs;
    Matrice * derived_inputs;
    Matrice * outputs;
public:
    Activaion_Function();
    ~Activaion_Function();

    virtual void forward(Matrice * inputs) {
        throw invalid_argument("Unimplemented");
    }
    virtual void backward(Matrice * derived_inputs) {
        throw invalid_argument("Unimplemented");
    }

    virtual Matrice * predictions(Matrice * outputs) const {
        throw invalid_argument("Unimplemented");
    }

    Matrice * get_outputs() const;
    Matrice * get_derived_inputs() const;

    Matrice * apply_function(Matrice * data, double (*func)(double)) const;
};


class Linear_Activation : public Activaion_Function 
{
public:
    Matrice * predictions(Matrice * outputs) const override;

    void forward(Matrice * inputs) override;
    void backward(Matrice * derived_inputs) override;
};

class Sigmoid_Activation : public Activaion_Function 
{
public:
    Matrice * predictions(Matrice * outputs) const override;

    void forward(Matrice * inputs) override;
    void backward(Matrice * derived_inputs) override;
};

class ReLU_Activation : public Activaion_Function 
{
public:
    Matrice * predictions(Matrice * outputs) const override;

    void forward(Matrice * inputs) override;
    void backward(Matrice * derived_inputs) override;
};

class Softmax_Activation : public Activaion_Function 
{
public:
    Matrice * predictions(Matrice * outputs) const override;

    void forward(Matrice * inputs) override;
    void backward(Matrice * derived_inputs) override;
};

double sigmoid_function(double x);

double sigmoid_function_derivative(double x);

double ReLU_function(double x);

double ReLU_function_derivative(double x);

#endif