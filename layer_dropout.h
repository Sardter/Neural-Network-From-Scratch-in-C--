#ifndef __LAYER_DROPOUT
#define __LAYER_DROPOUT

#include "layer.h"
#include "matrice_lib.h"
#include "activaion_functions.h"

class Layer_Dropout : public Layer
{
private:
    Matrice * binary_mask;
    
    double rate;
public:
    Layer_Dropout(double rate, Activaion_Function * function);
    ~Layer_Dropout();

    void forward(Matrice * inputs, bool is_training) override;
    void backward(Matrice * derivated_inputs) override;

    double get_rate() const;
    void set_rate(double rate);
};

#endif