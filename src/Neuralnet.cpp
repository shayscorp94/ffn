/*
 * net.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: oliv
 */


#include <iostream>
#include <armadillo>
#include <algorithm> // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
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
//  Data processing
	const int nassets{487};
	const int nlines{756};
	const double end_train{600};
	dataframe Data{756,nassets,"cleanIndex.csv"};
	mat Train = Data.getData().rows(0,end_train+0);


	vector<double (*)(const double &)> fs{opt::I,opt::Lrelu,opt::Lrelu,opt::Lrelu};
	vector<double (*)(const double &)> ds{opt::One,opt::DLrelu,opt::DLrelu,opt::DLrelu};


	vector<int> layers{487,250,125,1};
//	Net has only one vector of coefficients but several vecotrs of nodes values so that for
//	each sample we have a Network
 	Net N(layers,fs,ds,/* number of samples : */end_train-9,/*number of threads : */4);

//	Network initialization
	for(int d = 0 ; d != end_train+1-10 ; ++d){
	//		For each available date, we calculate a gradient and then we average
		N.n(d,0,0) = 0; /* N.n(sample,layer,node) */
		N.setTarget(d) = Train(d+10,0); /* target corresponding do the d th sample*/
		for(int s = 1 ; s != nassets ; ++s){
			N.n(d,0,s) = Train(d,s);
		}
		N.update(d); /* update for the network corresponding to sample d */

	}
//
	dataframe init(N.getNcoeffs(),1,"Coeffs590assets_batchSize26_9.csv");
	N.get_coeffs() = init.getData();

	mat stats = opt::stochastic_descent(&N,/*learning rate*/1e-8,/*stoping condition*/0.01,/* batch size*/26,/*linear search enabled*/true,/*number of epochs*/1000,/* minlr*/1e-10);


	opt::result(&N);
	dataframe S{stats,vector<std::string>{"gradient norm","error"}};
	S.write_csv("Stats590assets_batchSize26_10.csv");

	dataframe coeffs{N.get_coeffs(),vector<std::string>{"coeffs"}};
	coeffs.write_csv("Coeffs590assets_batchSize26_10.csv");



	return 0;
}


