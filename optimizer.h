#ifndef __OPTIMIZER
#define __OPTIMIZER

#include "layer_dens.h"
#include "matrice_lib.h"
#include <vector>

using namespace std;

class Optimizer
{
protected:
    double learning_rate;
public:
    Optimizer(double learning_rate);
    ~Optimizer();

    void update_params(Layer_Dens * layer);
};

class Stochastic_Gradient_Descent: public Optimizer
{
private:
    double current_learning_rate;
    double decay;
    long long iterations;

    double momentum;
    Matrice * weight_momentum;
    Vector * bias_momentum;
public:
    Stochastic_Gradient_Descent(double learning_rate = 1, double decay = 0, double momentum = 0);
    ~Stochastic_Gradient_Descent();

    void pre_update();
    void post_update();
    void update_params(Layer_Dens * layer);
};


#endif