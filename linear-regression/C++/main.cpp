#include "linear_regression.h"
#include <cstdlib>

int main() {
    dataset ds('+');
    vector<vector<double> > trainSet = ds.getTrainSet();
    vector<double> trainValues = ds.getTrainValues();
    dataset test('*');
    vector<vector<double> > testSet = test.getTestSet();
    //vector<double> testValues = ds.getTestValues();
    linearRegression LR(trainSet, trainValues, 0.1, testSet);

    system("pause");

    return 0;
}