#include "matrice_lib.h"
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include <numeric>

using namespace std;

Vector::Vector(size_t size, double default_value) 
{
    this->data = vector<double>(size, default_value);
}

Vector::Vector(vector<double> data) 
{
    this->data = data;
}

Vector::Vector(double data[], size_t size) 
{
    this->data = vector<double>(data, data + size);
}

Vector::Vector(int data[], size_t size) 
{
    this->data = vector<double>(data, data + size);
}

Vector * Vector::copy() const
{
    return new Vector(this->data);
}

double Vector::size() const
{
    return this->data.size();
}

Vector * Vector::product(double scalar) const
{
    Vector * product = this->copy();
    for (size_t i = 0; i < this->data.size(); i++)
    {
        product->data[i] *= scalar;
    }
    return product;
}

Vector * Vector::product(Vector * v) const
{
    Vector * product = this->copy();
    for (size_t i = 0; i < this->data.size(); i++)
    {
        product->data[i] *= v->data[i];
    }
    return product;
}

double Vector::dot_product(Vector * v) const
{
    int n = this->size(), m = v->size();
    if (n != m)
        throw invalid_argument("Vectors must be the same size: a ->" + to_string(n) + " b -> " + to_string(m));

    double product = 0;
    for (size_t i = 0; i < this->size(); i++)
    {
        product += this->data[i] * v->data[i];
    }
    return product;   
}

Vector * Vector::add(Vector * v) const
{
    Vector * sum = this->copy();
    for (size_t i = 0; i < this->data.size(); i++)
    {
        sum->data[i] += v->data[i];
    }
    return sum;
}

Vector * Vector::subtract(Vector * v) const
{
    Vector * sub = this->copy();
    for (size_t i = 0; i < this->data.size(); i++)
    {
        sub->data[i] -= v->data[i];
    }
    return sub;
}

double Vector::sum() const
{
    return accumulate(this->data.begin(), this->data.end(), 0);
}

Vector * Vector::abs() const
{
    Vector * res = this->copy();
    for (size_t i = 0; i < this->data.size(); i++)
    {
        res->data[i] = this->data[i] < 0 ? this->data[i] * -1 : this->data[i];
    }
    return res;
}

//////////////////////////////////////////////////////////////////////////////////////////////////


size_t Matrice::column_size() const
{
    return this->data.size();
}

size_t Matrice::row_size() const
{
    return this->data[0].size();
}

Matrice::Matrice(size_t column_size, size_t row_size, double default_value)
{
    this->data = vector<vector<double> >(column_size, vector<double>(row_size, default_value));
}

Matrice::Matrice(vector<vector<double> > data)
{
    this->data = data;
}

Matrice::Matrice(vector<double> data)
{
    this->data = vector<vector<double> >(1, data);
}

Vector * Matrice::get_column(int index) const
{
    Vector * column = new Vector(this->column_size());
    for (size_t i = 0; i < this->column_size(); i++)
    {
        column->data[i] = this->data[i][index];
    }
    return column;
}

Vector * Matrice::get_row(int index) const
{
    return new Vector(this->data[index]);
}


bool Matrice::equal_shape(Matrice * m) const 
{
    return this->column_size() == m->column_size() && this->row_size() == m->row_size();
}

Matrice * Matrice::copy() const 
{
    return new Matrice(this->data);
}

void _check_size(vector<double> &a, vector<double> &b)
{
    int n = a.size(), m = b.size();
    if (n != m)
        throw invalid_argument("Vectors must be the same size: a ->" + to_string(n) + " b -> " + to_string(m));
}

void _check_size(Vector * a, Vector  * b)
{
    int n = a->data.size(), m = b->data.size();
    if (n != m)
        throw invalid_argument("Vectors must be the same size: a ->" + to_string(n) + " b -> " + to_string(m));
}

Matrice * Matrice::add(Vector * v) const
{
    Matrice * res = this->copy();
    for (size_t i = 0; i < res->column_size(); i++)
    {
        res->data[i] = v->add(res->get_row(i))->data;
    }

    return res;   
}

Matrice * Matrice::add(Matrice * m) const
{
    Matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] += m->data[i][j];
        }
        
    }
    return res;
}

Matrice * Matrice::add(double x) const
{
    Matrice * res = this->copy();
    for (size_t i = 0; i < res->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] += x;
        }
    }

    return res;   
}

Matrice * Matrice::operator+(Vector * v) const
{
    return this->add(v);
}

Matrice * Matrice::dot_product(Matrice * m) const
{
    size_t a_row = this->row_size();
    size_t a_column = this->column_size();
    size_t b_row = m->row_size();
    size_t b_column = m->column_size();

    if (a_row != b_column)
        throw invalid_argument("Sizes are not equal: a->" + to_string(a_column) + " b->" + to_string(b_row));

    Matrice * res = new Matrice(a_column, b_row);


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

Matrice * Matrice::product(double x) const
{
    Matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] *= x;
        }
    }
    return res;
}

Matrice * Matrice::product(Matrice * m) const
{
    Matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] *= m->data[i][j];
        } 
    }
    return res;
}

double Matrice::sum() const
{
    double sum = 0;
    for (size_t i = 0; i < this->column_size(); i++)
    {
        sum += accumulate(this->data[i].begin(), this->data[i].end(), 0);
    }
    
    return sum;
}

Matrice * Matrice::abs() const
{
    Matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] = res->data[i][j] < 0 ? res->data[i][j] * -1 : res->data[i][j];
        } 
    }
    return res;
}

Matrice * Matrice::operator * (Matrice * m) const 
{
    return this->dot_product(m);
}

Matrice * Matrice::division(Matrice * m) const
{
    Matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] /= m->data[i][j];
        }
        
    }
    return res;
}

Matrice * Matrice::subtract(Matrice * m) const
{
    Matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] -= m->data[i][j];
        }
        
    }
    return res;
}

Matrice * Matrice::division(double x) const
{
    Matrice * res = this->copy();
    for (size_t i = 0; i < this->column_size(); i++)
    {
        for (size_t j = 0; j < this->row_size(); j++)
        {
            res->data[i][j] /= x;
        }
        
    }
    return res;
}

Matrice * Matrice::transpose() const
{
    size_t col = this->column_size();
    size_t row = this->row_size();


    Matrice *res = new Matrice(row, col);

    for (size_t i = 0; i < col; ++i)
    {
        for (size_t j = 0; j < row; ++j)
        {
            res->data[j][i] = this->data[i][j];
        }
    }

    return res;
}

Matrice * Matrice::operator ! () const 
{
    return this->transpose();
}

Vector * Matrice::columns_sum() const 
{
    size_t col_size = this->column_size();
    size_t row_size = this->row_size();
    Vector * res = new Vector(col_size);

    for (size_t i = 0; i < col_size; i++)
    {
        for (size_t j = 0; j < row_size; j++)
        {
            res->data[j] += this->data[i][j];
        }
    }
    
    return res;
}

Vector * Matrice::rows_sum() const 
{
    size_t col_size = this->column_size();
    size_t row_size = this->row_size();
    Vector * res = new Vector(row_size);

    for (size_t i = 0; i < col_size; i++)
    {
        for (size_t j = 0; j < row_size; j++)
        {
            res->data[i] += this->data[i][j];
        }
    }
    
    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Matrice * discrete_to_one_hot(Vector * nums, size_t limit) {
    size_t row_num = nums->size();
    Matrice * res = new Matrice(limit, row_num);
    for (size_t i = 0; i < row_num; i++)
    {
        res->data[nums->data[i]][i] = 1;
    }
    return res->transpose();
}


Matrice * diag_flat(vector<double> * nums) {
    size_t size = nums->size();
    Matrice * res = new Matrice(size, size);
    for (size_t i = 0; i < size; i++)
    {
        res->data[i][i] = nums->at(i);
    }
    return res;
}

Matrice * jacobian_matrice(vector<double> * input) {
    Matrice * m = new Matrice(*input);

    return diag_flat(input)->subtract(m->transpose()->dot_product(m));
}

double standard_deviation(Matrice * m) {
    size_t row_size = m->row_size();
    size_t col_size = m->column_size();

    double mean = m->sum() / (row_size * col_size);
    double variance = 0;
    for (size_t i = 0; i < col_size; i++)
    {
        for (size_t j = 0; j < row_size; j++)
        {
            variance += pow(m->data[i][j] - mean, 2);
        }
    }
    return sqrt(variance / (row_size * col_size));
}

ostream& operator << (ostream& stream, Matrice m) {
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

ostream& operator << (ostream& stream, Vector m) {
    for (double j : m.data)
    {
        stream << j << " ";
    }
    return stream << endl;
}