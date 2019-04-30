#include "LR.h"

LR::LR()
{
}

LR::~LR()
{
}

void LR::TrainAlgorithm(int iterations, double m, double b)
{
	int iter = 0;
	m = _m;
	b = _b;

	if (!IsConverged() && iter < iterations)
	{
		//calculate m
		//hyper parameters
		double step = 2 / double(iter + 2);

		//local minima is the smallest point in the graph 
		double m_grad = 0; // slope
		double b_intercept = 0; //y-intercept

		for (int i = 0; i < _numElems; i++)
		{
			m_grad += _xVals[i] * ((m * _xVals[i] + b) - _yVals[i]);
		}
		m_grad = (2 * m_grad) / _numElems;
		
		//calculate y intercept
		for (int i = 0; i < _numElems; i++)
		{
			b_intercept = ((m * _xVals[i] + b) - _yVals[i]);
		}
		b_intercept = (2 * b_intercept) / _numElems;

		m = m - (step * m_grad);
		b = b - (step * b_intercept);

		std::cout << "m:\t" << m << ", b:\t" << b << "\r\n";
		std::cout << "grad_m:\t" << m_grad << ", grad_b:\t" << b_intercept << "\r\n";
		iter++;
	}	
}



double LR::Regress(double x_)
{
	//draws line of best fit y = m x+c
	double res = _m * x_ + _b;
	return res;
}

bool LR::IsConverged()
{
	double error = 0;
	double threshold = 0.01;

	for (int i = 0; i < _numElems; i++)
	{
		//y = m x + b^2 
		error += ((_m * _xVals[i] + _b) - _yVals[i]) * ((_m * _xVals[i] + _b) - _yVals[i]);
	}

	error /= _numElems;
	std::cout << "Error " << error << "\r\n";
	bool res = (abs(error) > _oldErr - threshold && abs(error) < _oldErr + threshold) ? true : false;
	_oldErr = abs(error);
	return res;
}
