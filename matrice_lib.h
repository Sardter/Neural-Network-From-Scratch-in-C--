#ifndef __MATRICE_LIBARARY
#define __MATRICE_LIBARARY

#include <vector>
#include <iostream>

using namespace std;

struct Vector {
    vector<double> data;

    Vector(size_t size, double default_value = 0);

    Vector(vector<double> data);

    Vector(double data[], size_t size);

    Vector * copy() const;

    Vector * product(double scalar) const;

    Vector * product(Vector * v) const;

    double dot_product(Vector * v) const;

    Vector * add(Vector * v) const;

    Vector * subtract(Vector * v) const;

    Vector * abs() const;

    double sum() const;

    double size() const;
};


struct Matrice
{
    vector<vector<double> > data;

    Vector * get_column(int index) const;
    Vector * get_row(int index) const;

    size_t row_size() const;
    size_t column_size() const;

    Matrice(size_t column_size, size_t row_size, double default_value = 0);

    Matrice(vector<vector<double> > data);

    Matrice(vector<double> data);

    bool equal_shape(Matrice * m) const;

    Matrice * copy() const;

    Matrice * add(Vector * v) const;

    Matrice * add(Matrice * m) const;

    Matrice * add(double x) const;

    Matrice * subtract(Matrice * m) const;

    Matrice * dot_product(Matrice * m) const;

    double sum() const;

    Matrice * abs() const;

    Matrice * product(Matrice * m) const;

    Matrice * product(double x) const;

    Matrice * division(Matrice * m) const;

    Matrice * division(double x) const;

    Matrice * transpose() const;

    Vector * columns_sum() const;

    Vector * rows_sum() const;

    Matrice *  operator + (Vector * v) const;

    Matrice * operator * (Matrice * m) const;

    Matrice * operator ! () const;
};

ostream& operator << (ostream& stream, Matrice m);

ostream& operator << (ostream& stream, vector<double> m);

ostream& operator << (ostream& stream, Vector m);

Matrice * discrete_to_one_hot(vector<int> * nums, size_t limit);

Matrice * diag_flat(vector<double> * nums);

Matrice * jacobian_matrice(vector<double> * input);

double standard_deviation(Matrice * m);
#endif