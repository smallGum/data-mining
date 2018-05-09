#include "linear_regression.h"

int main() {
    dataset ds('1');
    vector<vector<double> > trainSet = ds.getTrainSet();
    vector<double> trainValues = ds.getTrainValues();
    //vector<vector<double> > testSet = ds.getTestSet();
    //vector<double> testValues = ds.getTestValues();
    linearRegression LR(trainSet, trainValues, 1000.0);

    return 0;
}