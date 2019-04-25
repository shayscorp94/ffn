/*
 * opt.h
 *
 *  Created on: Apr 12, 2019
 *      Author: oliv
 *
 *      Static Class containing several functions for the Net
 */

#ifndef OPT_H_
#define OPT_H_

#include "math.h"
#include "Net.h"
#include <algorithm> // std::random_shuffle
#include "dataframe.h"



namespace net {

class opt {
public:
//	Activation functions and their derivatives
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

// computes the gradient of Net for a given sample number and target
static void grad(const Net & N,const int sample,Net & G,const double & target);
// perfroms gradient descent for the Net
static arma::vec grad_descent(Net * N, const double & etha/*initial learning rate*/,const double & eps/*stoping condition*/, bool linSearch = true);
// calculates gradient for samples start to end (end excluded) enables parallelization
// 1st argument :ptr to mat of grads, only the column corresponding to the thread number is used to avoid data races
static void partial_grad(arma::mat * thread_grads, Net * N,std::vector<Net> * Gs,const int start,const int end,const int thread );
// performs gradient on all samples, will use parallelization (n threads specified in construction of Net)
static void gradient( Net * N,std::vector<Net> * Gs,arma::vec * res_grad,arma::mat * thread_grads);
// calculates error for samples start to end (end excluded) enables parallelization
static void partial_err(double * res, Net * N,const int start,const int end,const int thread );
// computes error on all samples, will use parallelization (n threads specified in construction of Net)
static double err( Net * N); /* Updates N before computing the error : useful for linear search */
// displays the prevision versus target value
static void result( Net * N,std::string str = "None");

// similar as grad_descent exept that we use a vec [0, ... , nsamples-1] that we permute at the begining of each epoch
// to makes sure batches are made of shuffled samples
static arma::mat stochastic_descent(Net * N, const double & etha,const double & eps,const int batchSize=1, bool linSearch = true,const int & nepochs =100,const double & minlr = 1e-9);
static arma::mat stochastic_descent_adam(Net * N, const double & etha,const double & eps,const int batchSize=1, bool linSearch = true,const int & nepochs =100,const double & minlr = 1e-9);

static void gradient_st( Net * N,std::vector<Net> * Gs,arma::vec * res_grad,arma::mat * thread_grads,const arma::vec * ,const int batchStart,const int batchEnd);

static void partial_grad_st(arma::mat * thread_grads, Net * N,std::vector<Net> * Gs,const arma::vec * sigma,const int start,const int end,const int thread );

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
