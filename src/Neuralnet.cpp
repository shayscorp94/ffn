/*
 * net.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: oliv
 */


#include <iostream>
#include <armadillo>
#include <algorithm>    // std::random_shuffle
#include <random>
#include <thread>

#include "Net.h"
#include "opt.h"
#include "math.h"
#include "dataframe.h"

using namespace std;
using namespace arma;
using namespace vSpace;
using namespace net;



int main(){

	const int nassets{487};
	const int nlines{756};
	const double end_train{10};
	dataframe Data{756,nassets,"cleanIndex.csv"};
	mat Train = Data.getData().rows(0,end_train+0);


	vector<double (*)(const double &)> fs{opt::I,opt::Lrelu,opt::Lrelu,opt::Lrelu};
	vector<double (*)(const double &)> ds{opt::One,opt::DLrelu,opt::DLrelu,opt::DLrelu};


	vector<int> layers{487,250,125,1};

	Net N(layers,fs,1);

	for(int d = 0 ; d != end_train+1-10 ; ++d){
	//		For each available date, we calculate a gradient and then we average
		N.n(d,0,0) = 0;
		N.setTarget(d) = Train(d+10,0);
		for(int s = 1 /* do not use the current index price*/ ; s != nassets ; ++s){
			N.n(d,0,s) = Train(d,s);
		}
		N.update(d);
	}
	cout << N.v(0,3,0);



	opt::grad_descent(&N,1e-8,0.01);


	return 0;
}


