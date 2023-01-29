#ifndef __MODEL
#define __MODEL

#include <vector>
#include "layer.h"
#include "loss_functions.h"
#include "layer_tools.h"
#include "optimizer.h"
#include "matrice_lib.h"
#include <iostream>

using namespace std;

class Model
{
private:
    vector<Layer> * layers;

    Loss_Function * loss;
    Optimizer * optimizer;
    Accuracy * accuracy;

    Layer * input_layer;
    Layer * output_layer;
public:
    Model();
    Model(vector<Layer> layers);
    Model(Layer layers[], size_t size);
    ~Model();

    void add_layer(Layer const layer);

    Layer& get_layer(size_t index) const;

    Layer& operator [] (size_t index) const;

    size_t size() const;

    void set_loss_function(Loss_Function * loss);

    void set_optimizer(Optimizer * optimizer);

    void set_accuracy(Accuracy * accuracy);

    void train(Matrice * X, Matrice * y, size_t epoches = 1, size_t print_ratio = 1);

    void train(Matrice * X, Vector * y, size_t epoches = 1, size_t print_ratio = 1);

    void predict(Matrice * X, Matrice * y);

    void predict(Matrice * X, Vector * y);

    void prepare();

    Matrice * forward(Matrice * inputs, bool is_training, double& regularization_loss);

    void backward(Matrice * output, Matrice * y);
};

ostream& operator << (ostream& stream, Model& m);

#endif