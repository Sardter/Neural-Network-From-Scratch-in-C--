#ifndef __LAYER_TOOLS
#define __LAYER_TOOLS

#include "matrice_lib.h"
#include "activaion_functions.h"
#include "loss_functions.h"

using namespace std;

class Activation_Softmax_Loss_Catogorical_Crossentropy {
private:
    Matrice * output;
    Matrice * derived_inputs;
    Softmax_Activation activation;
    categorical_cross_entropy loss;
public:
    Activation_Softmax_Loss_Catogorical_Crossentropy();
    ~Activation_Softmax_Loss_Catogorical_Crossentropy();

    double forward(Matrice * inputs, Vector * targets);
    void backward(Matrice * inputs, Vector * targets);

    Matrice * get_output() const;
    Matrice * get_derived_inputs() const;
};

class Accuracy {
public:
    Accuracy();
    ~Accuracy();

    virtual void prepare(Vector * targets, bool reprepare);
    virtual void prepare(Matrice * targets, bool reprepare);

    virtual bool compare(Matrice * data, Vector * targets, int i);
    virtual bool compare(Matrice * data, Matrice * targets, int i, int j);

    virtual double calculate(Matrice * data, Vector * targets);
    virtual double calculate(Matrice * data, Matrice * targets);
};

class Accuracy_Regression : public Accuracy
{
private:
    double percision;
public:
    Accuracy_Regression();
    ~Accuracy_Regression();

    void prepare(Matrice * targets, bool reprepare) override;

    bool compare(Matrice * data, Matrice * targets, int i, int j) override;
};

class Accuracy_Classification : public Accuracy
{
public:
    Accuracy_Classification();
    ~Accuracy_Classification();

    bool compare(Matrice * data, Vector * targets, int i) override;

    bool compare(Matrice * data, Matrice * targets, int i, int j) override;

    double calculate(Matrice * data, Matrice * targets) override;
};

Matrice * generate_gaugian_weights(int columns, int row, double rate = 0.1);

Matrice * generate_binomial_weights(int columns, int row, double rate);

Vector * default_biases(int neurons);

double calculate_clasification_accuracy(Matrice * data, Vector * targets);

double calculate_regression_accuracy(Matrice * data, Matrice * targets);

double accuracy_percision(Matrice * data, double divisor = 1);

#endif