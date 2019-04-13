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

static void grad(const Net & N,Net & G,const double & target);

static arma::mat grad_descent(const arma::mat & v0, arma::mat (& grad) (const arma::mat *), const double & etha,const double & eps);

inline arma::mat g(const arma::mat * v);

static void update_partial(Net * N, const int l,const int start,const int end);

static void update(Net * N);


private:
	opt();
	virtual ~opt();
};

} /* namespace net */

#endif /* OPT_H_ */
