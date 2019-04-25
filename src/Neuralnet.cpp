
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


int main() {
	//  Data processing
	const int nassets{ 485 };
	const int nlines{ 756 };
	const double end_train{ 600 };
	dataframe Data{ 756,nassets,"cleanIndex.csv" };
	mat Train = Data.getData().rows(0, end_train + 0);

	const int start_test = 591;
	const int end_test = 750;

	mat Test = Data.getData().rows(start_test, end_test);


	vector<double(*)(const double &)> fs{ opt::I,opt::Lrelu,opt::Lrelu,opt::Lrelu };
	vector<double(*)(const double &)> ds{ opt::One,opt::DLrelu,opt::DLrelu,opt::DLrelu };


	vector<int> layers{ 484, 242, 121,1 };
	//	Net has only one vector of coefficients but several vecotrs of nodes values so that for
	//	each sample we have a Network

	//	Network initialization Train
	Net N(layers, fs, ds,/* number of samples : */end_train - 9,/*number of threads : */4);

	for (int d = 0; d != end_train + 1 - 10; ++d) {
		//		For each available date, we calculate a gradient and then we average
		N.n(d, 0, 0) = 0; /* N.n(sample,layer,node) */
		N.setTarget(d) = Train(d + 10, 0); /* target corresponding do the d th sample*/
		for (int s = 1; s != nassets; ++s) {
			N.n(d, 0, s) = Train(d, s);
		}
		N.update(d); /* update for the network corresponding to sample d */

	}

	// 		Network initialization Test

	//	 	Net N(layers,fs,ds,/* number of samples : */end_test-start_test-9,/*number of threads : */4);
	// 		for(int d = 0 ; d != end_test-start_test-9 ; ++d){
	// 		//		For each available date, we calculate a gradient and then we average
	// 			N.n(d,0,0) = 0; /* N.n(sample,layer,node) */
	//
	// 			N.setTarget(d) = Test(d+10,0); /* target corresponding do the d th sample*/
	//
	// 			for(int s = 1 ; s != nassets ; ++s){
	// 				N.n(d,0,s) = Test(d,s);
	// 			}
	// 			N.update(d); /* update for the network corresponding to sample d */
	//
	// 		}

	//	dataframe init(N.getNcoeffs(),1,"Coeffs590assets_batchSize26_11.csv");
	//	N.get_coeffs() = init.getData();

	mat stats = opt::stochastic_descent_adam(&N,/*learning rate*/1e-8,/*stoping condition*/0.01,/* batch size*/20,/*linear search enabled*/false,/*number of epochs*/10,/* minlr*/1e-10);

	opt::err(&N);
	opt::result(&N);
	//	dataframe S{stats,vector<std::string>{"gradient norm","error"}};
	//	S.write_csv("Stats590assets_batchSize26_11.csv");
	//
	//	dataframe coeffs{N.get_coeffs(),vector<std::string>{"coeffs"}};
	//	coeffs.write_csv("Coeffs590assets_batchSize26_11.csv");



	return 0;
}


