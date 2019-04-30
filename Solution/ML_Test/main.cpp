#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <numeric>
#include <cmath>
#include <math.h>
#include <limits>
#include <string>
//K Means
#include <utility>
#include <valarray>
#include <iterator>
#include <algorithm> // min_element

#include<iomanip>

using namespace std;

class LinearRegression
{
public:

	LinearRegression() {};
	~LinearRegression() {};
	LinearRegression(std::vector<double> & _x_vals_, std::vector<double> _y_vals_) : m_x_vals(_x_vals_), 
	m_y_vals(_y_vals_), m_num_elems(_y_vals_.size()), m_old_err(std::numeric_limits<double>::max()) {}

	void trainAlgorithm(int num_iters_, double m_init_, double b_init_)
	{
		int iter = 0;
		m_m = m_init_;
		m_b = b_init_;

		while (!IsConverged() && iter < num_iters_)
		{
			//hyper parameters
			double step = 0.0002;
			//double step = 2 / double(iter + 2);
			//double step = 0.002;
			//local minima is the smallest point in the graph 
			// y = m x + b
			double m_grad = 0; // slope
			double b_grad = 0; //y-intercept

			//partial derivative

			//compute gradient (- removed)
			for (int i = 0; i < m_num_elems; i++)
			{
				m_grad += m_x_vals[i] * ((m_m * m_x_vals[i] + m_b) - m_y_vals[i]);
			}
			m_grad = (2 * m_grad) / m_num_elems;

			//grad b
			for (int i = 0; i < m_num_elems; i++)
			{
				b_grad += ((m_m * m_x_vals[i] + m_b) - m_y_vals[i]);
			}
			b_grad = (2 * b_grad) / m_num_elems;

			//take step
			m_m = m_m - (step * m_grad);
			m_b = m_b - (step * b_grad);

			std::cout << "m:\t" << m_m << ", b:\t" << m_b << "\r\n";
			std::cout << "grad_m:\t" << m_grad << ", grad_b:\t" << b_grad << "\r\n";
			iter++;
		
		}

	}

	double regress(double x_)
	{
		//draws line of best fit 
		double res = m_m * x_ + m_b;
		return res;
	}


private:

	bool IsConverged()
	{
		double error = 0;
		double thresh = 0.01;

		for (unsigned i = 0; i < m_num_elems; i++)
		{
			//y = m x + b^2 
			error += ((m_m * m_x_vals[i] + m_b) - m_y_vals[i]) * ((m_m * m_x_vals[i] + m_b) - m_y_vals[i]);
		}

		error /= m_num_elems;
		std::cout << "Error " << error << "\r\n";
		bool res = (abs(error) > m_old_err - thresh && abs(error) < m_old_err + thresh) ? true : false;
		m_old_err = abs(error);
		return res;

	}

	std::vector<double> m_x_vals;
	std::vector<double> m_y_vals;
	double m_num_elems;
	double m_old_err;
	double m_m;
	double m_b;

};

class KMeans
{
public:
	KMeans() {};
	~KMeans() {};

	KMeans(int k, std::vector<pair< double,  double>> & data_) : m_k(k), m_centriod(k), m_data(k)
	{
		m_data[0] = data_; //this just assigns the first label to all data
	}

	void ClusterData(std::valarray<pair< double,  double>> & init_centroids, int num_iters)
	{
		m_centriod = init_centroids; 
		this->AssignLables();

		int i = 0;
		while(i < num_iters)
		{
			std::cout << "Running Iterator " << i << "\r\n";
			this->ComputeMeans();
			this->AssignLables();
			i++;
		}
	}

	void PrintClusters() const
	{
		for (int k = 0; k < m_k; k++) {
			cout << "Cluster: " << k << "\r\n";
			for (auto const & feature : m_data[k]) {
				std::cout << " [" << std::get<0>(feature) << "," << std::get<1>(feature) << "] ";
			}
			cout << "\r\n";
		}
	}
private:

	bool ComputeMeans()
	{
		bool res = true;

		//for each cluster/class, loop through data members;

		for (int i = 0; i < m_k; i++)
		{
			//Container for mean
			std::pair< double, double> mean(0, 0);
			//access length of vector of valarray container
			int num_features_of_k = m_data[i].size();
			//Iterate through data and add them together, increment data member/ sum them up
		
			for (auto const & it : m_data[i])
			{
				//add each value of x and y to the corresponding mean of x and y
				std::get<0>(mean) += std::get<0>(it);
 				std::get<1>(mean) += std::get<1>(it);
			}
			//divide mean by the number of features to get the average
			std::get<0>(mean) /= num_features_of_k;
			std::get<1>(mean) /= num_features_of_k;
			
			//check if mean and res are both true, check for convergence 
			res = (m_centriod[i] == mean && res == true) ? true : false;
			
			//Setting current mean to the overall mean
			m_centriod[i] = mean;
			std::cout << "Cluster Centroid " << i << ": " << std::get<0>(mean) << " , " << std::get<1>(mean) << "\t\n";;
			
		}
		return res;
	}


	int ComputeClosestCentroid(const std::pair<double,double> & point_)
	{
		std::valarray<double> distances(m_k);

		for (int i = 0; i < m_k; i++)
		{
			double del_x = std::get<0>(point_) - std::get<0>(m_centriod[i]);
			double del_y = std::get<1>(point_) - std::get<1>(m_centriod[i]);
			double dist = sqrt((del_x * del_x) + (del_y * del_y));
			distances[i] = dist;
		}
		
		auto closest_mean = distance(begin(distances), min_element(begin(distances), end(distances)));
		//std::cout << closest_mean << "THIS IS THE MEAN" << std::endl;
		return closest_mean;
	}

	void AssignLables()
	{
		//For each data points, assign to a cluster

		//set container for the vector of pairs
		std::valarray<std::vector<std::pair< double,double>>> new_data(m_k);
		//loop over array of clusters above
		for (auto const & clust : m_data)
		{
			for (auto const & feature : clust)
			{
				int closest_mean = this->ComputeClosestCentroid(feature);
				new_data[closest_mean].push_back(feature);
			}
		}
		m_data = new_data;
	}

	int m_k;
	int m_features;
	std::valarray<pair<double, double>> m_centriod; //container holds current centriod
	std::valarray<std::vector<std::pair<double, double>>> m_data; //array is of length and holds the vectors of the data points classified as that label
};

class NewK
{
public:
	NewK() {};
	~NewK() {};

	NewK(int k) : m_k(k)
	{
		cout << m_k;
	};

private:


	int m_k;
};

class PTest
{
public:
	PTest() {};
	~PTest() {};

	int ShowP() {};
private:

	int m_n = 5;
	NewK m_newK = NewK(5);

};
int main(int argc, char ** argv)
{

	/*std::vector<double> y({ 60.3, 67.8, 70.3, 82.1 });
	std::vector<double> x({ 1.67, 1.72, 1.77, 1.82, });*/

	/*std::vector<double> y({ 11, 12, 13, 14 });
	std::vector<double> x({ 40, 41, 42, 43, });

	LinearRegression lr(x, y);

	lr.trainAlgorithm(1000, 0, 0);
	std::cout << lr.regress(44) << std::endl;
*/
/*std::vector<std::pair<double, double>> data =
{ { 53.445107, -2.252369 },{ 53.443698, -2.254933 },{ 53.443860, -2.248530 },{ 53.462558, -2.253039 },{53.460236, -2.253401},
{ 53.460988, -2.249157 } };
*/

	NewK *newK = new NewK(5);

	//KMeans km(2, data);

	//std::valarray<std::pair<double, double>> init_centroids = { { 53.462878, -2.245696 },{ 53.443948, -2.256569 } };

	//km.ClusterData(init_centroids, 100);
	//
	//km.PrintClusters();


	return 1;
}

