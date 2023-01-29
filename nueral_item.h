#ifndef __NEURAL_ITEM
#define __NEURAL_ITEM

#include "matrice_lib.h"
#include <stdexcept>
#include <iostream>

struct Nueral_Item
{
    
    virtual void forward(Matrice * inputs) {
        throw invalid_argument("Tried calling forward from abstract neural item");
    }
    virtual void backward(Matrice * derivated_inputs) {
        throw invalid_argument("Tried calling backward from abstract neural item");
    }

    virtual Matrice * get_output() {
        return nullptr;
    }
};

#endif