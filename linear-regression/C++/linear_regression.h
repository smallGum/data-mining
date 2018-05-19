#ifndef _LINEAR_REGRESSION_
#define _LINEAR_REGRESSION_

#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

#define VAR_NUM 384
#define SAMPLE_NUM 5000
#define ALL_SAMPLE_NUM 25000
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

    if (whichSetForTest == '+') {
        ifstream in("newTrain.csv");
        for (int i = 0; i < ALL_SAMPLE_NUM; ++i) {
            vector<double> temp(VAR_NUM);
            double val;
            for (int j = 0; j < VAR_NUM; ++j) { in >> temp[j]; }
            in >> val;
            trainSet.push_back(temp);
            trainValues.push_back(val);
        }
        in.close();
    } else if (whichSetForTest == '*') {
        ifstream in("newTest.csv");
        for (int i = 0; i < ALL_SAMPLE_NUM; ++i) {
            vector<double> temp(VAR_NUM);
            double val;
            for (int j = 0; j < VAR_NUM; ++j) { in >> temp[j]; }
            testSet.push_back(temp);
        }
        in.close();
    } else {
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
    void predict(vector<vector<double> >& testSet);
    void saveResult(vector<double>& result);

public:
    linearRegression(vector<vector<double> >& trainSet, vector<double>& trainValues, double _lambda, vector<vector<double> >& testSet);
    vector<double>& getTheta() { return theta; }
    double getConstTheta() { return constTheta; }
};

linearRegression::linearRegression(vector<vector<double> >& trainSet, vector<double>& trainValues, double _lambda, vector<vector<double> >& testSet) {
    learningRate = 0.001;
    constTheta = 0.0;
    lambda = _lambda;
    theta = vector<double>(VAR_NUM, 0.0);
    predictValues = vector<double>(trainSet.size(), 0.0);

    findBestTheta(trainSet, trainValues);
    predict(testSet);
}

void linearRegression::findBestTheta(vector<vector<double> >& trainSet, vector<double>& trainValues) {
    double minMSRE = calMSRE(trainSet, trainValues);
    
    cout << "for alpha = " << learningRate << ", cost value = " << minMSRE << endl;

    while (learningRate > 1e-8) {
        vector<double> originalTheta(theta);
        double originalConstTheta = constTheta;
        updateTheta(trainSet, trainValues);
        double curMSRE = calMSRE(trainSet, trainValues);

        if (curMSRE > minMSRE || fabs(curMSRE - minMSRE) < 1e-5) {
            copy(originalTheta.begin(), originalTheta.end(), theta.begin());
            constTheta = originalConstTheta;
            curMSRE = calMSRE(trainSet, trainValues);
            learningRate /= RATE_DESCENT_SCALE;
        } else { minMSRE = curMSRE; }

        cout << "for alpha = " << learningRate << ", cost value = " << curMSRE << endl;
    }
}

void linearRegression::updateTheta(vector<vector<double> >& trainSet, vector<double>& trainValues) {
    for (int k = 0; k < theta.size(); ++k) {
        double newValue = 0.0;
        for (int i = 0; i < trainSet.size(); ++i) {
            newValue += (1.0 / (double)trainSet.size()) * (predictValues[i] - trainValues[i]) * trainSet[i][k];
        }
        newValue += (lambda / (double)trainSet.size()) * theta[k];
        theta[k] = theta[k] - learningRate * newValue;
    }

    double newValue = 0.0;
    for (int i = 0; i < trainSet.size(); ++i) {
        newValue += (1.0 / (double)trainSet.size()) * (predictValues[i] - trainValues[i]) * 1.0;
    }
    newValue += (lambda / (double)trainSet.size()) * constTheta;
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
    for (int j = 0; j < VAR_NUM; ++j) { MSRE += (lambda / (2.0 * (double)trainSet.size())) * theta[j] * theta[j]; }
    MSRE += (lambda / (2.0 * (double)trainSet.size())) * constTheta * constTheta;

    return MSRE;
}

void linearRegression::predict(vector<vector<double> >& testSet) {
    vector<double> results(testSet.size(), constTheta);
    for (int i = 0; i < testSet.size(); ++i) {
        for (int j = 0; j < VAR_NUM; ++j) {
            results[i] += theta[j] * testSet[i][j];
        }
    }

    saveResult(results);
}

void linearRegression::saveResult(vector<double>& result) {
    cout << "saving results..." << endl; 
    ofstream out("result.csv");
    out << "id," << "reference" << endl;
    for (int i = 0; i < result.size(); ++i) {
        out << i << "," << result[i] << endl;
    }
    out.close();
}

#endif