#ifndef __LAYER_DROPOUT
#define __LAYER_DROPOUT

#include "layer.h"
#include "matrice_lib.h"

class Layer_Dropout : public Layer
{
private:
    double rate;
public:
    Layer_Dropout(double rate);
    ~Layer_Dropout();

    void forward(Matrice * inputs);
    void backward(Matrice * derivated_inputs);

    double get_rate() const;
    void set_rate(double rate);
};

#endif