#ifndef __LAYER_DENSE
#define __LAYER_DENSE

#include "layer.h"
#include "matrice_lib.h"
#include "activaion_functions.h"
#include "vector"

class Layer_Dens: public Layer
{
private:
    Vector * biases;
    Vector * biases_derivative;
    Matrice * weights;
    Matrice * weights_derivative;

    // weight regularizer
    double WR_L1, WR_L2;
    // bias regularizer
    double BR_L1, BR_L2;
public:
    Layer_Dens(
        int input_num, int neoron_num, 
        Activaion_Function * function,
        double WR_L1 = 0, double WR_L2 = 0, 
        double BR_L1 = 0, double BR_L2 = 0
    );
    Layer_Dens(Matrice * weights, Vector * biases, 
        Activaion_Function * function,
        double WR_L1 = 0, double WR_L2 = 0, 
        double BR_L1 = 0, double BR_L2 = 0
    );
    ~Layer_Dens();

    void forward(Matrice * inputs, bool is_training) override;
    void backward(Matrice * derivated_inputs) override;

    Vector * get_biases() const;
    Vector * get_biases_derivatives() const;
    Matrice * get_weights() const;
    Matrice * get_weight_derivatives() const;

    double get_regularization_loss() const override;

    void set_biases(Vector * v);
    void set_weights(Matrice * m);

    bool is_trainable() const override;
};


#endif