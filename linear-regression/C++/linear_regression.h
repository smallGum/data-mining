#ifndef _LINEAR_REGRESSION_
#define _LINEAR_REGRESSION_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

#define VAR_NUM 384
#define SAMPLE_NUM 50000
#define RATE_DESCENT_SCALE 2.0

class dataset {
private:
    char whichSetForTest;
    vector<vector<double> > trainSet;
    vector<double> trainValues;
    vector<vector<double> > testSet;
    vector<double> testValues;

public:
    dataset(char which);
    vector<vector<double> >& getTrainSet() { return trainSet; }
    vector<vector<double> >& getTestSet() { return testSet; }
    vector<double>& getTrainValues() { return trainValues; }
    vector<double>& getTestValues() { return testValues; }
};

dataset::dataset(char which) {
    whichSetForTest = which;

    for (char c = '1'; c <= '5'; ++c) {
        string filename("5_folds_data/part");
        filename.push_back(c);
        filename += ".csv";
        ifstream in(filename);

        if (c != whichSetForTest) {
            for (int i = 0; i < SAMPLE_NUM; ++i) {
                vector<double> temp(VAR_NUM);
                double val;
                for (int j = 0; j < VAR_NUM; ++j) { in >> temp[j]; }
                in >> val;
                trainSet.push_back(temp);
                trainValues.push_back(val);
            }
        } else {
            for (int i = 0; i < SAMPLE_NUM; ++i) {
                vector<double> temp(VAR_NUM);
                double val;
                for (int j = 0; j < VAR_NUM; ++j) { in >> temp[j]; }
                in >> val;
                testSet.push_back(temp);
                testValues.push_back(val);
            }
        }

        in.close();
    }
}

class linearRegression {
private:
    double learningRate;
    double constTheta;
    double lambda;
    vector<double> theta;
    vector<double> predictValues;

    void findBestTheta(vector<vector<double> >& trainSet, vector<double>& trainValues);
    void updateTheta(vector<vector<double> >& trainSet, vector<double>& trainValues);
    double calMSRE(vector<vector<double> >& trainSet, vector<double>& trainValues);

public:
    linearRegression(vector<vector<double> >& trainSet, vector<double>& trainValues, double _lambda);
};

linearRegression::linearRegression(vector<vector<double> >& trainSet, vector<double>& trainValues, double _lambda) {
    learningRate = 0.02;
    constTheta = 0.0;
    lambda = _lambda;
    theta = vector<double>(VAR_NUM, 0.0);
    predictValues = vector<double>(trainSet.size(), 0.0);

    findBestTheta(trainSet, trainValues);
}

void linearRegression::findBestTheta(vector<vector<double> >& trainSet, vector<double>& trainValues) {
    double lastMSRE = calMSRE(trainSet, trainValues);
    
    cout << "for alpha = " << learningRate << ", MSRE = " << lastMSRE << endl;

    int cnt = 0;
    while (true) {
        updateTheta(trainSet, trainValues);
        double curMSRE = calMSRE(trainSet, trainValues);

        cout << "for alpha = " << learningRate << ", MSRE = " << curMSRE << endl;
        
        if (curMSRE > lastMSRE || fabs(curMSRE - lastMSRE) < 1e-10) { learningRate /= RATE_DESCENT_SCALE; }
        if (learningRate < 1e-6) { break; }

        lastMSRE = curMSRE;
        cnt++;
    }
}

void linearRegression::updateTheta(vector<vector<double> >& trainSet, vector<double>& trainValues) {
    vector<double> originalTheta(theta);
    for (int k = 0; k < theta.size(); ++k) {
        double newValue = 0.0;
        for (int i = 0; i < trainSet.size(); ++i) {
            newValue += (1.0 / (double)trainSet.size()) * (predictValues[i] - trainValues[i]) * trainSet[i][k];
        }
        newValue += (1.0 / (double)VAR_NUM) * originalTheta[k];
        theta[k] = theta[k] - learningRate * newValue;
    }

    double newValue = 0.0;
    for (int i = 0; i < trainSet.size(); ++i) {
        newValue += (1.0 / (double)trainSet.size()) * (predictValues[i] - trainValues[i]);
    }
    constTheta = constTheta - learningRate * newValue;

}

double linearRegression::calMSRE(vector<vector<double> >& trainSet, vector<double>& trainValues) {
    double MSRE = 0.0;

    for (int i = 0; i < trainSet.size(); ++i) {
        double predictValue = constTheta;
        for (int j = 0; j < VAR_NUM; ++j) { predictValue += trainSet[i][j] * theta[j]; }
        predictValues[i] = predictValue;
        MSRE += (1.0 / (2.0 * (double)trainSet.size())) * (predictValue - trainValues[i]) * (predictValue - trainValues[i]);
    }
    for (int j = 0; j < VAR_NUM; ++j) { MSRE += (lambda / (2.0 * double(VAR_NUM))) * theta[j] * theta[j]; }

    return MSRE;
}

#endif