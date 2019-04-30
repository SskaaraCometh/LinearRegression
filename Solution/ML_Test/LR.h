#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <numeric>
#include <cmath>
#include <math.h>
#include <limits>

class LR
{
public:
	LR();
	~LR();
	LR(std::vector<double> x, std::vector<double> y) : _xVals(x), _yVals(y), _numElems(y.size()), _oldErr(std::numeric_limits<double>::max()) {};
	void TrainAlgorithm(int iterations, double m, double b);

	double Regress(double x_);

private:
	bool IsConverged();

	std::vector<double> _xVals;
	std::vector<double> _yVals;
	double _numElems;
	double _oldErr;
	double _m;
	double _b;
};

