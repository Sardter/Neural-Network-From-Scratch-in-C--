#include <iostream>

#include "matrice_lib.h"

using namespace std;

int main()
{
    double bias[] = {2, 3, 0.5};
    vector<double> biases(begin(bias), end(bias));

    double w0[] = {0.2, 0.8, -0.5, 1.0};
    vector<double> v0(begin(w0), end(w0));
    double w1[] = {0.5, -0.91, 0.26, -0.5};
    vector<double> v1(begin(w1), end(w1));
    double w2[] = {-0.26, -0.27, 0.17, 0.87};
    vector<double> v2(begin(w2), end(w2));

    vector<double> weight[] = {v0, v1, v2};
    vector<vector<double> > weights(begin(weight), end(weight));
    Matrice weights_matrice = Matrice(weights);

    double i0[] = {1, 2, 3, 2.5};
    vector<double> iv0(begin(i0), end(i0));
    double i1[] = {2, 5, -1, 2};
    vector<double> iv1(begin(i1), end(i1));
    double i2[] = {-1.5, 2.7, 3.3, -0.8};
    vector<double> iv2(begin(i2), end(i2));

    vector<double> input[] = {iv0, iv1, iv2};
    vector<vector<double> > inputs(begin(input), end(input));
    Matrice inputs_matrice = Matrice(inputs);

    /* matrice * out = vector_addition(
        matrice_multiplication(&inputs_matrice, 
            matrice_transpose(&weights_matrice))
    , &biases); */

    Matrice * out = inputs_matrice.dot_product(weights_matrice.transpose())->add(new Vector(biases));
    cout << "output: " << endl;
    cout << *out << endl;

    cout << "column 1: " << *out->get_column(1) << endl;
    cout << "row 1: " << *out->get_row(1) << endl;

    cout << "columns sum: " << *out->columns_sum() << endl;
    cout << "rows sum: " << *out->rows_sum() << endl;

    int i3[] = {1, 3, 3, 4};
    Vector iv3(begin(i3), end(i3));
    cout << *discrete_to_one_hot(&iv3, 6) << endl;

    cout << *diag_flat(&iv1) << endl;

    double i4[] = {.7, .1, .2};
    vector<double> iv4(begin(i4), end(i4));
    cout << *jacobian_matrice(&iv4) << endl;
    cout << *jacobian_matrice(&iv4)->dot_product((new Matrice(diag_flat(&iv4)->get_row(0)->data))->transpose()) << endl;

    return 0;
}