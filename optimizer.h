#ifndef __OPTIMIZER
#define __OPTIMIZER

#include "layer_dens.h"
#include "matrice_lib.h"
#include <vector>

using namespace std;

class optimizer
{
protected:
    double learning_rate;
public:
    optimizer(double learning_rate);
    ~optimizer();

    void update_params(layer_dens * layer);
};

class stochastic_gradient_descent: public optimizer
{
private:
    double current_learning_rate;
    double decay;
    long long iterations;

    double momentum;
    matrice * weight_momentum;
    vector<double> * bias_momentum;
public:
    stochastic_gradient_descent(double learning_rate = 1, double decay = 0, double momentum = 0);
    ~stochastic_gradient_descent();

    void pre_update();
    void post_update();
    void update_params(layer_dens * layer);
};


#endif