#ifndef __LOSS
#define __LOSS

#include "matrice_lib.h"
#include "layer_dens.h"
#include "nueral_item.h"
#include <vector>
#include <stdexcept>

using namespace std;

class Loss_Function
{
protected:
    Matrice * inputs_derivatives;
public:
    Loss_Function();
    ~Loss_Function();

    double loss_percentage(Matrice * input, Vector * target);

    double loss_percentage(Matrice * input, Matrice * target);

    virtual Vector * forward(Matrice * input, Vector * target) {
        throw invalid_argument("Unimplemented");
    }

    virtual Vector * forward(Matrice * input, Matrice * target) {
        throw invalid_argument("Unimplemented");
    }

    virtual void backward(Matrice * input, Vector * target) {
        throw invalid_argument("Unimplemented");
    }

    virtual void backward(Matrice * input, Matrice * target) {
        throw invalid_argument("Unimplemented");
    }

    Matrice * get_input_derivatives() const;
};

class categorical_cross_entropy : public Loss_Function {
public:
    Vector * forward(Matrice * input, Vector * target) override;

    Vector * forward(Matrice * input, Matrice * target) override;

    void backward(Matrice * input, Vector * target) override;

    void backward(Matrice * input, Matrice * target) override;
};

class binary_cross_entropy : public Loss_Function {
public:
    //Vector * forward(Matrice * input, Vector * target);

    Vector * forward(Matrice * input, Matrice * target) override;

    //void backward(Matrice * input, Vector * target);

    void backward(Matrice * input, Matrice * target) override;
};

class mean_squared_error : public Loss_Function {
public:
    //Vector * forward(Matrice * input, Vector * target);

    Vector * forward(Matrice * input, Matrice * target) override;

    //void backward(Matrice * input, Vector * target);

    void backward(Matrice * input, Matrice * target) override;
};


#endif