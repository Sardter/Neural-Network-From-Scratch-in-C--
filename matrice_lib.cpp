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

matrice::matrice(vector<double> data)
{
    this->data = vector<vector<double> >(1, data);
}

vector<double> * matrice::get_column(int index) const
{
    vector<double> * column = new vector<double>(this->column_size(), 0);
    for (size_t i = 0; i < this->column_size(); i++)
    {
        column->at(i) = this->data[i][index];
    }
    return column;
}

vector<double> * matrice::get_row(int index) const
{
    return new vector<double>(this->data[index]);
}

bool matrice::equal_shape(matrice * m) const 
{
    return this->column_size() == m->column_size() && this->row_size() == m->row_size();
}

matrice * matrice::copy() const 
{
    return new matrice(this->data);
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

vector<double> * scalar_product(vector<double> * a, double b) {
    vector<double> *product = new vector<double>(*a);
    for (size_t i = 0; i < a->size(); i++)
    {
        product->at(i) *= b;
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

vector<double> *vector_subtraction(vector<double> * a, vector<double> * b)
{
    _check_size(*a, *b);

    for (size_t i = 0; i < (*a).size(); i++)
    {
        (*a)[i] -= (*b)[i];
    }
    return a;
}

matrice * matrice::add(vector<double> * v) const
{
    matrice * res = this->copy();
    for (size_t i = 0; i < res->column_size(); i++)
    {
        res->data[i] = *vector_addition(&(res->data[i]), v);
    }

    return res;   
}

matrice * matrice::add(matrice * m) const
{
    matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] += m->data[i][j];
        }
        
    }
    return res;
}

matrice * matrice::operator+(vector<double> * v) const
{
    return this->add(v);
}

matrice * matrice::product(matrice * m) const
{
    size_t a_row = this->row_size();
    size_t a_column = this->column_size();
    size_t b_row = m->row_size();
    size_t b_column = m->column_size();

    if (a_row != b_column)
        throw invalid_argument("Sizes are not equal: a->" + to_string(a_column) + " b->" + to_string(b_row));

    matrice * res = new matrice(a_column, b_row);


    for (size_t i = 0; i < a_column; i++)
    {
        for (size_t j = 0; j < b_row; j++)
        {
            for (size_t k = 0; k < a_row; k++)
            {
                res->data[i][j] += this->data[i][k] * m->data[k][j];
            }
        }
    }

    return res;
}

matrice * matrice::product(double x) const
{
    matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] *= x;
        }
        
    }
    return res;
}

matrice * matrice::operator * (matrice * m) const 
{
    return this->product(m);
}

matrice * matrice::division(matrice * m) const
{
    matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] /= m->data[i][j];
        }
        
    }
    return res;
}

matrice * matrice::subtract(matrice * m) const
{
    matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] -= m->data[i][j];
        }
        
    }
    return res;
}

matrice * matrice::division(double x) const
{
    matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] /= x;
        }
        
    }
    return res;
}

matrice * matrice::transpose() const
{
    size_t col = this->column_size();
    size_t row = this->row_size();


    matrice *res = new matrice(row, col);

    for (size_t i = 0; i < col; ++i)
    {
        for (size_t j = 0; j < row; ++j)
        {
            res->data[j][i] = this->data[i][j];
        }
    }

    return res;
}

matrice * matrice::operator ! () const 
{
    return this->transpose();
}

vector<double> * matrice::columns_sum() const 
{
    size_t col_size = this->column_size();
    size_t row_size = this->row_size();
    vector<double> * res = new vector<double>(row_size, 0);

    for (size_t i = 0; i < col_size; i++)
    {
        for (size_t j = 0; j < row_size; j++)
        {
            res->at(j) += this->data[i][j];
        }
    }
    
    return res;
}

vector<double> * matrice::rows_sum() const 
{
    size_t col_size = this->column_size();
    size_t row_size = this->row_size();
    vector<double> * res = new vector<double>(col_size, 0);

    for (size_t i = 0; i < col_size; i++)
    {
        for (size_t j = 0; j < row_size; j++)
        {
            res->at(i) += this->data[i][j];
        }
    }
    
    return res;
}

matrice * discrete_to_one_hot(vector<int> * nums, size_t limit) {
    size_t row_num = nums->size();
    matrice * res = new matrice(limit, row_num);
    for (size_t i = 0; i < row_num; i++)
    {
        res->data[nums->at(i)][i] = 1;
    }
    return res->transpose();
}


matrice * diag_flat(vector<double> * nums) {
    size_t size = nums->size();
    matrice * res = new matrice(size, size);
    for (size_t i = 0; i < size; i++)
    {
        res->data[i][i] = nums->at(i);
    }
    return res;
}

matrice * jacobian_matrice(vector<double> * input) {
    matrice * m = new matrice(*input);

    return diag_flat(input)->subtract(m->transpose()->product(m));
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