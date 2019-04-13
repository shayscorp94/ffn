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



	vector<double (*)(const double &)> fs{opt::I,opt::Lrelu,opt::Lrelu};
	vector<double (*)(const double &)> ds{opt::One,opt::DLrelu,opt::DLrelu};
	vector<int> layers{4,2,1};

	Net N(layers,fs,1);


	opt::grad_descent(&N,0.01,0.01);


	return 0;
}


