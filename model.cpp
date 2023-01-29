#include "model.h"
#include <vector>
#include <iostream>
#include "layer.h"
#include "matrice_lib.h"
#include "layer_tools.h"

using namespace std;

Model::Model() 
{
    this->layers = new vector<Layer>();
    this->optimizer = nullptr;
    this->accuracy = nullptr;
    this->loss = nullptr;
}

Model::Model(vector<Layer> layers)
{
    this->layers = new vector<Layer>(layers);
}

Model::Model(Layer layers[], size_t size)
{
    this->layers = new vector<Layer>(size);
    for (size_t i = 0; i < size; i++)
    {
        this->layers->at(i) = layers[i];
    }
}

Model::~Model() 
{
    delete this->layers;
    if (this->optimizer != nullptr) delete this->optimizer;
    if (this->accuracy != nullptr) delete this->accuracy;
    if (this->loss != nullptr) delete this->loss;
}

void Model::add_layer(Layer const layer) 
{
    this->layers->push_back(layer);
}

Layer& Model::get_layer(size_t index) const
{
    return this->layers->at(index);
}

Layer& Model::operator [] (size_t index) const
{
    return this->get_layer(index);
}

size_t Model::size() const
{
    return this->layers->size();
}

void Model::set_loss_function(Loss_Function * loss)
{
    this->loss = loss;   
}

void Model::set_optimizer(Optimizer * optimizer)
{
    this->optimizer = optimizer;
}

void Model::set_accuracy(Accuracy * accuracy) {
    this->accuracy = accuracy;
}

void Model::prepare()
{
    this->input_layer = new Layer(nullptr);

    this->output_layer = &this->layers->at(this->size() - 1);
}

Matrice * Model::forward(Matrice * inputs, bool is_training ,double& regularizatio_loss)
{
    this->input_layer->forward(inputs, is_training);
    Matrice * out = this->input_layer->get_output();

    for (size_t i = 0; i < this->size(); i++)
    {
        this->layers->at(i).forward(out, is_training);
        regularizatio_loss += this->layers->at(i).get_regularization_loss();
        out = this->layers->at(i).get_output();
    }
    
    return out;
}

void Model::backward(Matrice * outputs, Matrice * y)
{
    this->loss->backward(outputs, y);
    Matrice * derived = this->loss->get_input_derivatives();

    for (size_t i = this->size(); i >= 0; i--)
    {
        this->layers->at(i).backward(derived);
        derived = this->layers->at(i).get_inputs_derivatives();
    }
}

void Model::train(Matrice * X, Matrice * y, size_t epoches = 1, size_t print_ratio = 1) {
    this->accuracy->prepare(y, false);
    for (size_t i = 0; i < epoches; i++)
    {
        double regularization_loss = 0;
        Matrice * out = this->forward(X, true, regularization_loss);
        double data_loss = this->loss->loss_percentage(out, y);

        double total_loss = regularization_loss + data_loss;

        Matrice * predictions = this->output_layer->get_activation()->predictions(out);
        double accuracy = this->accuracy->calculate(predictions, y);

        this->backward(out, y);

        this->optimizer->pre_update();
        for (size_t i = 0; i < this->size(); i++)
        {
            if (this->layers->at(i).is_trainable()) {
                Layer_Dens * dens = dynamic_cast<Layer_Dens *> (&this->layers->at(i));
                this->optimizer->update_params(dens);
            }
        }
        this->optimizer->post_update();

        if (epoches % print_ratio == 0) {
            cout << endl;
            cout << "Epoch: " << i << endl;
            cout << "Accuracy: " << accuracy << endl;
            cout << "Data Loss: " << data_loss << endl;
            cout << "Regularization Loss: " << regularization_loss << endl;
            cout << "Learning Rate: " << this->optimizer->get_learning_rate() << endl;
            cout << endl;
        }
    } 
}

void Model::predict(Matrice * X, Matrice * y) {
    this->accuracy->prepare(y, false);
    double regularization_loss = 0;
    Matrice * out = this->forward(X, false, regularization_loss);
    double data_loss = this->loss->loss_percentage(out, y);

    double total_loss = regularization_loss + data_loss;

    Matrice * predictions = this->output_layer->get_activation()->predictions(out);
    double accuracy = this->accuracy->calculate(predictions, y);

    cout << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << "Data Loss: " << data_loss << endl;
    cout << "Regularization Loss: " << regularization_loss << endl;
    cout << endl;
}

ostream& operator << (ostream& stream, Model& m)
{
    stream << "//////////// Model ////////////" << endl;
    for (size_t i = 0; i < m.size(); i++)
    {
        stream << "Layer " << i << endl;
        stream << m[i] << endl;
    }
    return stream;
}