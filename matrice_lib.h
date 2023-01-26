#ifndef __MATRICE_LIBARARY
#define __MATRICE_LIBARARY

#include <vector>
#include <iostream>

using namespace std;

struct matrice
{
    vector<vector<double> > data;

    vector<double> * get_column(int index) const;
    vector<double> * get_row(int index) const;

    size_t row_size() const;
    size_t column_size() const;

    matrice(size_t column_size, size_t row_size);

    matrice(vector<vector<double> > data);

    matrice(vector<double> data);

    bool equal_shape(matrice * m) const;

    matrice * copy() const;

    matrice * add(vector<double> * v) const;

    matrice * add(matrice * m) const;

    matrice * subtract(matrice * m) const;

    matrice * product(matrice * m) const;

    matrice * product(double x) const;

    matrice * division(matrice * m) const;

    matrice * division(double x) const;

    matrice * transpose() const;

    vector<double> * columns_sum() const;

    vector<double> * rows_sum() const;

    matrice *  operator + (vector<double> * v) const;

    matrice * operator * (matrice * m) const;

    matrice * operator ! () const;
};

double dot_product(vector<double> * a, vector<double> * b);

vector<double> * scalar_product(vector<double> * a, double b);

vector<double> *vector_addition(vector<double> * a, vector<double> * b);

vector<double> *dot_product(vector<double> &a, vector<vector<double> > &b);

ostream& operator << (ostream& stream, matrice m);

ostream& operator << (ostream& stream, vector<double> m);

matrice * discrete_to_one_hot(vector<int> * nums, size_t limit);

matrice * diag_flat(vector<double> * nums);

matrice * jacobian_matrice(vector<double> * input);
#endif