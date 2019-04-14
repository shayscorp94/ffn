/*
 * opt.h
 *
 *  Created on: Apr 12, 2019
 *      Author: oliv
 */

#ifndef OPT_H_
#define OPT_H_

#include "math.h"
#include "Net.h"
#include <algorithm> // std::random_shuffle



namespace net {

class opt {
public:
static double relu(const double & d){
		return d > 0 ? d : 0;
	}
static double Lrelu(const double & d){
		return d > 0 ? d : 0.01*d;
	}
static double DLrelu(const double & d){
		return d > 0 ? 1 : 0.01;
	}
static double I(const double & t){
		return t;
	}
static double One(const double & t){
		return 1;
	}
static double Drelu(const double & d){
		return d > 0 ? 1 : 0;
	}
static double sm(const double & t){
		return (t>10)?t:log(1+exp(t));
	}
static double Dsm(const double & t){



		return (t>100)?1:exp(t)/(1+exp(t));
	}

static void grad(const Net & N,const int sample,Net & G,const double & target);

static arma::vec grad_descent(Net * N, const double & etha,const double & eps, bool linSearch = true);

static void partial_grad(arma::vec * res, Net * N,std::vector<Net> * Gs,const int start,const int end,const int thread );

static void gradient( Net * N,std::vector<Net> * Gs,arma::vec * res_grad);

static void partial_err(double * res, Net * N,const int start,const int end,const int thread );

static double err( Net * N); /* Updates N before computing the error : useful for linear search */

static void result( Net * N);

static arma::vec stochastic_descent(Net * N, const double & etha,const double & eps,const int batchSize=1, bool linSearch = true);

static void gradient_st( Net * N,std::vector<Net> * Gs,arma::vec * res_grad,const arma::vec * ,const int batchStart,const int batchEnd);

static void partial_grad_st(arma::vec * res, Net * N,std::vector<Net> * Gs,const arma::vec * sigma,const int start,const int end,const int thread );

static double err_st(Net * N,const arma::vec * sigma,const int batchStart,const int batchEnd);

static void partial_err_st(double * res, Net * N,const arma::vec * sigma,const int start,const int end,const int thread );

///static void update_partial(Net * N, const int l,const int start,const int end);
//
//static void update(Net * N);


private:
	opt();
	virtual ~opt();

};

} /* namespace net */

#endif /* OPT_H_ */
