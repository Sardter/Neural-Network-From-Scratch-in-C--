#ifndef __LAYER_DROPOUT
#define __LAYER_DROPOUT

#include "layer.h"
#include "matrice_lib.h"

class layer_dropout : public layer
{
private:
    double rate;
public:
    layer_dropout(double rate);
    ~layer_dropout();

    void forward(Matrice * inputs);
    void backward(Matrice * derivated_inputs);

    double get_rate() const;
    void set_rate(double rate);
};

#endif