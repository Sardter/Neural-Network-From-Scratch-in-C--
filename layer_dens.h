#ifndef __LAYER_DENSE
#define __LAYER_DENSE

#include "matrice_lib.h"
#include "vector"

class layer_dens
{
private:
    matrice * weights;
    vector<double> * biases;
    matrice * output;
public:
    layer_dens(int input_num, int neoron_num);
    ~layer_dens();

    void forward(matrice * inputs);
    matrice * get_weights() const;
    vector<double> * get_biases() const;
    matrice * get_output() const;
};


#endif