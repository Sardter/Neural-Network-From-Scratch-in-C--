#ifndef __LOSS
#define __LOSS

#include "matrice_lib.h"
#include <vector>
#include <stdexcept>

using namespace std;

class loss_function {
protected:
    matrice * inputs_derivatives;
public:
    loss_function();
    ~loss_function();

    double loss_percentage(matrice * input, vector<int> * target);

    double loss_percentage(matrice * input, matrice * target);

    vector<double> * forward(matrice * input, vector<int> * target) {
        throw invalid_argument("Unimplemented");
    }

    vector<double> * forward(matrice * input, matrice * target) {
        throw invalid_argument("Unimplemented");
    }

    void backward(matrice * input, vector<int> * target) {
        throw invalid_argument("Unimplemented");
    }

    void backward(matrice * input, matrice * target) {
        throw invalid_argument("Unimplemented");
    }
};

class categorical_cross_entropy : public loss_function {
public:
    vector<double> * forward(matrice * input, vector<int> * target);

    vector<double> * forward(matrice * input, matrice * target);

    void backward(matrice * input, vector<int> * target);

    void backward(matrice * input, matrice * target);
};


#endif