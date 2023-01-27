#ifndef __LOSS
#define __LOSS

#include "matrice_lib.h"
#include "layer_dens.h"
#include <vector>
#include <stdexcept>

using namespace std;

class Loss_Function {
protected:
    Matrice * inputs_derivatives;
public:
    Loss_Function();
    ~Loss_Function();

    double loss_percentage(Matrice * input, vector<int> * target);

    double loss_percentage(Matrice * input, Matrice * target);

    Vector * forward(Matrice * input, vector<int> * target) {
        throw invalid_argument("Unimplemented");
    }

    Vector * forward(Matrice * input, Matrice * target) {
        throw invalid_argument("Unimplemented");
    }

    void backward(Matrice * input, vector<int> * target) {
        throw invalid_argument("Unimplemented");
    }

    void backward(Matrice * input, Matrice * target) {
        throw invalid_argument("Unimplemented");
    }

    double regularization_loss(Layer_Dens * layer) const;
};

class categorical_cross_entropy : public Loss_Function {
public:
    Vector * forward(Matrice * input, vector<int> * target);

    Vector * forward(Matrice * input, Matrice * target);

    void backward(Matrice * input, vector<int> * target);

    void backward(Matrice * input, Matrice * target);
};

class binary_cross_entropy : public Loss_Function {
public:
    //Vector * forward(Matrice * input, vector<int> * target);

    Vector * forward(Matrice * input, Matrice * target);

    //void backward(Matrice * input, vector<int> * target);

    void backward(Matrice * input, Matrice * target);
};

class mean_squared_error : public Loss_Function {
public:
    //Vector * forward(Matrice * input, vector<int> * target);

    Vector * forward(Matrice * input, Matrice * target);

    //void backward(Matrice * input, vector<int> * target);

    void backward(Matrice * input, Matrice * target);
};


#endif