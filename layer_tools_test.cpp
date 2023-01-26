#include <iostream>
#include "layer_tools.h"
#include "matrice_lib.h"

using namespace std;

matrice * soft_max_outputs() {
    double i0[] = {0.7, 0.1, 0.2};
    vector<double> iv0(begin(i0), end(i0));
    double i1[] = {0.1, 0.5, 0.4};
    vector<double> iv1(begin(i1), end(i1));
    double i2[] = {0.02, 0.9, 0.08};
    vector<double> iv2(begin(i2), end(i2));

    vector<double> input[] = {iv0, iv1, iv2};
    vector<vector<double> > inputs(begin(input), end(input));
    return new matrice(inputs);
}

vector<int> * targets() {
    int i[] = {0, 1, 1};
    return new vector<int>(begin(i), end(i));
}

int main() {
    activation_softmax_loss_catogorical_crossentropy sl;
    sl.backward(soft_max_outputs(), targets());
    cout << *sl.get_derived_inputs() << endl;
}