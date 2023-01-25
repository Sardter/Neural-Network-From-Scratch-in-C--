#ifndef __MATRICE_LIBARARY
#define __MATRICE_LIBARARY

#include <vector>
#include <iostream>

using namespace std;

struct matrice
{
    vector<vector<double> > data;

    vector<double> get_column(int index) const;
    vector<double> get_row(int index) const;

    size_t row_size() const;
    size_t column_size() const;

    matrice(size_t column_size, size_t row_size);
    matrice(vector<vector<double> > data);

    bool equal_shape(matrice * m) const;
};

double dot_product(vector<double> * a, vector<double> * b);

vector<double> *vector_addition(vector<double> &a, vector<double> &b);

matrice *vector_addition(matrice * a, vector<double> * b);

vector<double> *dot_product(vector<double> &a, vector<vector<double> > &b);

matrice *matrice_multiplication(matrice* a, matrice* b);

matrice *matrice_transpose(matrice* a);

ostream& operator << (ostream& stream, matrice m);

ostream& operator << (ostream& stream, vector<double> m);

#endif