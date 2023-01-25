#include <iostream>
#include "layer_dens.h"
#include "matrice_lib.h"
#include "activaion_functions.h"
#include "loss_functions.h"
#include "layer_tools.h"
#include <vector>

using namespace std;

matrice * inputs1() {
    double i0[] = {1, 2, 3, 2.5};
    vector<double> iv0(begin(i0), end(i0));
    double i1[] = {2, 5, -1, 2};
    vector<double> iv1(begin(i1), end(i1));
    double i2[] = {-1.5, 2.7, 3.3, -0.8};
    vector<double> iv2(begin(i2), end(i2));

    vector<double> input[] = {iv0, iv1, iv2};
    vector<vector<double> > inputs(begin(input), end(input));
    return new matrice(inputs);
}

matrice * inputs2() {
    double i0[] = {4.8, 1.21, 2.385};
    vector<double> iv0(begin(i0), end(i0));
    double i1[] = {8.9, -1.81, 0.2};
    vector<double> iv1(begin(i1), end(i1));
    double i2[] = {1.41, 1.051, 0.026};
    vector<double> iv2(begin(i2), end(i2));

    vector<double> input[] = {iv0, iv1, iv2};
    vector<vector<double> > inputs(begin(input), end(input));
    return new matrice(inputs);
}

vector<int> * inputs2_target() {
    double i0[] = {0, 0, 1};
    return new vector<int>(begin(i0), end(i0));
}

int main() {
    matrice * inputs_matrice = inputs1();

    layer_dens l1 = layer_dens(4, 5);
    l1.forward(inputs_matrice);
    cout << "layer 1 out:" << endl;
    cout << *l1.get_output() << endl;

    layer_dens l2 = layer_dens(5, 2);
    l2.forward(l1.get_output());

    cout << "layer 2 out:" << endl;
    cout << *l2.get_output() << endl;
    
    cout << "layer 2 out with activatioon:" << endl;
    cout << *apply_activation(l2.get_output(), rectifeid_linear) << endl;

    cout << "softmax: " << endl;
    matrice * soft = soft_max_activation(inputs2());
    cout << * soft << endl;

    cout << "loss: " << endl;
    cout << categorical_cross_entropy(soft, inputs2_target()) << endl;

    cout << "accuracy: " << endl;
    cout << calculate_accuracy(soft, inputs2_target()) << endl;

    return 0;
}