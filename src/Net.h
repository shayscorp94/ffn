/*
 * Net.h
 *
 *  Created on: Apr 12, 2019
 *      Author: oliv
 */

#ifndef NET_H_
#define NET_H_

#include <vector>
#include <armadillo>

namespace net {

class Net {
public:
	Net(const std::vector<int> & layers, std::vector<double (*)(const double &)> fs,std::vector<double (*)(const double &)> ds,const int & n_samples,const int & nthreads = 1 ); /* input a vector with the sizes of the layers, starting with the largest layer : l1 , l2 , lk */
	double & c(const int & layer, const int & start, const int & end); /* gives access to a coefficient, starting from the layer given as input, enables it to be modified ,DOES NOT UPDATE THE NODE VALUES  */
	double c(const int & layer, const int & start, const int & end)const; /* gives access to a coefficient, starting from the layer given as input, cannot be modified  */
	double & v(const int & sample,const int & layer, const int & i); /* gives access to a the f(node value) from the layer given as input, enables it to be modified, DOES NOT UPDATE THE NODE VALUES */
	double v(const int & sample,const int & layer, const int & i) const; /* gives access to the f(node value) from the layer given as input, cannot modified*/
	double & n(const int & sample,const int & layer, const int & i); /* gives access to a the node value from the layer given as input, enables it to be modified, DOES NOT UPDATE THE NODE VALUES */
	double n(const int & sample,const int & layer, const int & i) const; /* gives access to the node value from the layer given as input, cannot modified*/



	arma::vec & get_coeffs(); /* returns reference to the whole vector of coefficients to be used for the gradient descent */
	void update(const int & sample); /* updates node values given the current coeffs and node values of layer 0 */
	void print(const int & sample);
	virtual ~Net();

	const std::vector<int>& L() const {
		return layers;
	}

	const std::vector<double (*)(const double&)>& getFs() const {
		return fs;
	}

	const std::vector<double (*)(const double&)>& getDs() const {
		return ds;
	}

	int getNthreads() const {
		return nthreads;
	}
	double getTarget(const int & sample) const{
		return targets(sample);
	}
	double& setTarget(const int & sample){
		return targets(sample);
	}
	const double getNsamples() const { return nsamples;};
	const double getNcoeffs() const {return coeffs.n_rows; }

private:
const int nsamples;
const int nthreads;
arma::vec coeffs; /* coefficients : vector of size  ( l1*l2 + l2*l3 + ... + l{k-1}*lk ) */
std::vector<arma::vec> nodevals; /*contains the values at each node  f(ax+by+...), size is (l1+l2+...+lk) */
std::vector<arma::vec> nodes; /*contains the input at each node : ax+by+..., size is (l1+l2+...+lk) */
arma::vec targets;
std::vector<int> layers;
std::vector<double (*)(const double &)> fs;
std::vector<double (*)(const double &)> ds;







};

} /* namespace net */

#endif /* NET_H_ */
