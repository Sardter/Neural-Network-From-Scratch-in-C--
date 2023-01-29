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

    virtual void pre_update();
    virtual void post_update();
    virtual void update_params(Layer_Dens * layer);

    virtual double get_learning_rate() const;
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

    void pre_update() override;
    void post_update() override;
    void update_params(Layer_Dens * layer) override;

    double get_learning_rate() const override;
};


#endif