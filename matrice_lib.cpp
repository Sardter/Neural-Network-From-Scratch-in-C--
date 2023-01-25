#include "matrice_lib.h"
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>

using namespace std;

size_t matrice::column_size() const
{
    return this->data.size();
}

size_t matrice::row_size() const
{
    return this->data[0].size();
}

matrice::matrice(size_t column_size, size_t row_size)
{
    this->data = vector<vector<double> >(column_size, vector<double>(row_size, 0));
}

matrice::matrice(vector<vector<double> > data)
{
    this->data = data;
}

vector<double> matrice::get_column(int index) const
{
    vector<double> column = vector<double>(this->column_size(), 0);
    for (size_t i = 0; i < this->column_size(); i++)
    {
        column[i] = this->data[i][index];
    }
    return column;
}

vector<double> matrice::get_row(int index) const
{
    return this->data[index];
}

bool matrice::equal_shape(matrice * m) const 
{
    return this->column_size() == m->column_size() && this->row_size() == m->row_size();
}

void _check_size(vector<double> &a, vector<double> &b)
{
    int n = a.size(), m = b.size();
    if (n != m)
        throw invalid_argument("Vectors must be the same size: a ->" + to_string(n) + " b -> " + to_string(m));
}

double dot_product(vector<double> &a, vector<double> &b)
{
    _check_size(a, b);

    double product = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        product += a[i] * b[i];
    }
    return product;
}

vector<double> *dot_product(vector<double> &a, vector<vector<double> > &b)
{
    vector<double> *product = new vector<double>;

    for (size_t i = 0; i < b.size(); i++)
    {
        product->push_back(dot_product(a, b[i]));
    }

    return product;
}

vector<double> *vector_addition(vector<double> * a, vector<double> * b)
{
    _check_size(*a, *b);

    for (size_t i = 0; i < (*a).size(); i++)
    {
        (*a)[i] += (*b)[i];
    }
    return a;
}

matrice *vector_addition(matrice * a, vector<double> * b)
{
    for (size_t i = 0; i < a->column_size(); i++)
    {
        a->data[i] = *vector_addition(&(a->data[i]), b);
    }

    return a;   
}

matrice *matrice_multiplication(matrice* a, matrice* b)
{
    size_t a_row = a->row_size();
    size_t a_column = a->column_size();
    size_t b_row = b->row_size();
    size_t b_column = b->column_size();

    if (a_row != b_column)
        throw invalid_argument("Sizes are not equal: a->" + to_string(a_column) + " b->" + to_string(b_row));

    matrice *res = new matrice(a_column, b_row);


    for (size_t i = 0; i < a_column; i++)
    {
        for (size_t j = 0; j < b_row; j++)
        {
            for (size_t k = 0; k < a_row; k++)
            {
                res->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }

    return res;
}

matrice *matrice_transpose(matrice* a)
{
    size_t col = a->column_size();
    size_t row = a->row_size();


    matrice *res = new matrice(row, col);

    for (size_t i = 0; i < col; ++i)
    {
        for (size_t j = 0; j < row; ++j)
        {
            res->data[j][i] = a->data[i][j];
        }
    }

    return res;
}

ostream& operator << (ostream& stream, matrice m) {
    for (vector<double> i : m.data)
    {
        for (double j : i)
        {
            stream << j << " ";
        }
        stream << endl;
    }
    return stream;
}

ostream& operator << (ostream& stream, vector<double> m) {
    for (double j : m)
    {
        stream << j << " ";
    }
    return stream << endl;
}