#include <iostream>
#include "optimizer.h"
#include "matrice_lib.h"
#include "layer_dens.h"
#include "activaion_functions.h"
#include <vector>


using namespace std;

Matrice * inputs1() {
    double i0[] = {1, 2, 3, 2.5};
    vector<double> iv0(begin(i0), end(i0));
    double i1[] = {2, 5, -1, 2};
    vector<double> iv1(begin(i1), end(i1));
    double i2[] = {-1.5, 2.7, 3.3, -0.8};
    vector<double> iv2(begin(i2), end(i2));

    vector<double> input[] = {iv0, iv1, iv2};
    vector<vector<double> > inputs(begin(input), end(input));
    return new Matrice(inputs);
}

Matrice * inputs2() {
    double i0[] = {4.8, 1.21, 2.385};
    vector<double> iv0(begin(i0), end(i0));
    double i1[] = {8.9, -1.81, 0.2};
    vector<double> iv1(begin(i1), end(i1));
    double i2[] = {1.41, 1.051, 0.026};
    vector<double> iv2(begin(i2), end(i2));

    vector<double> input[] = {iv0, iv1, iv2};
    vector<vector<double> > inputs(begin(input), end(input));
    return new Matrice(inputs);
}

int main() {
    Stochastic_Gradient_Descent optimizer(0.001, 0.1, 0.5);
    Layer_Dens l1(4,5, new ReLU_Activation(), 0.01, 0.01, 0.01, 0.01);
    l1.forward(inputs1());

    for (size_t i = 1; i < 1000; i++)
    {
        l1.backward(l1.get_output());

        if (i % 10 == 0) {
            cout << *l1.get_weights() << endl;
            cout << *l1.get_biases() << endl;
        }
        

        optimizer.update_params(&l1);

        if (i % 10 == 0) {
            cout << *l1.get_weights() << endl;
            cout << *l1.get_biases() << endl;
        }

        l1.forward(l1.get_inputs());
    }
    
}