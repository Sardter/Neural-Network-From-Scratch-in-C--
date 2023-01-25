#include <iostream>
#include "layer_tools.h"
#include "matrice_lib.h"

using namespace std;

int main() {
    matrice w = *generate_rondom_weights(3, 4);
    cout << w << endl;
}