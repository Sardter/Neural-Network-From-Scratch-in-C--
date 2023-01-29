#include <iostream>
#include "layer_dens.h"
#include "matrice_lib.h"
#include "activaion_functions.h"
#include "loss_functions.h"
#include "layer_tools.h"
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

Matrice * weights() {
    double w0[] = {0.2, 0.8, -0.5, 1.0};
    vector<double> v0(begin(w0), end(w0));
    double w1[] = {0.5, -0.91, 0.26, -0.5};
    vector<double> v1(begin(w1), end(w1));
    double w2[] = {-0.26, -0.27, 0.17, 0.87};
    vector<double> v2(begin(w2), end(w2));

    vector<double> weight[] = {v0, v1, v2};
    vector<vector<double> > weights(begin(weight), end(weight));
    return (new Matrice(weights))->transpose();
}

vector<double> * biases() {
    double bias[] = {2, 3, 0.5};
    vector<double> biases(begin(bias), end(bias));

    return new vector<double>(biases);
}

Vector * inputs2_target() {
    double i0[] = {0, 0, 1};
    return new Vector(i0, 3);
}

int main() {
    Matrice * inputs_matrice = inputs1();
    Matrice * weights_matice = weights();
    vector<double> * bias = biases();

    //layer_dens l1 = layer_dens(weights_matice, bias);
    Layer_Dens l1 = Layer_Dens(4, 5);
    l1.forward(inputs_matrice);
    Matrice * l1_out = l1.get_output();
    cout << "layer 1 out:" << endl;
    cout << *l1_out << endl;

    /* ReLU_activation relu;
    relu.forward(l1_out);
    matrice * relu_out = relu.get_outputs();

    cout << "ReLU out: " << endl;
    cout << *relu_out << endl;

    relu.backward(relu_out);
    matrice * drelu = relu.get_derived_inputs();
    l1.backward(drelu);

    weights_matice = l1.get_weight_derivatives()->product(-0.001)->add(weights_matice);
    bias = vector_addition(scalar_product(l1.get_biases_derivatives(), -0.001), bias);

    cout << "Updated weights: " << endl;
    cout << *weights_matice << endl;
    cout << "Updated Biases: " << endl;
    cout << *bias << endl; */

    Layer_Dens l2 = Layer_Dens(5, 2);
    l2.forward(l1_out);

    cout << "layer 2 out:" << endl;
    cout << *l2.get_output() << endl;
    
    cout << "layer 2 out with activation:" << endl;
    ReLU_Activation relu;
    relu.forward(l2.get_output());
    cout << *relu.get_outputs() << endl;

    cout << "softmax: " << endl;
    Softmax_Activation soft_activation;
    soft_activation.forward(inputs2());
    Matrice * soft = soft_activation.get_outputs();
    cout << * soft << endl;

    soft_activation.backward(inputs2());
    Matrice * bsoft = soft_activation.get_derived_inputs();

    cout << * bsoft << endl;

    //cout << "loss: " << endl;
    //cout << categorical_cross_entropy(soft, inputs2_target()) << endl;

    cout << "accuracy: " << endl;
    cout << calculate_clasification_accuracy(soft, inputs2_target()) << endl;

    return 0;
}