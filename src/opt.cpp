/*
 * opt.cpp
 *
 *  Created on: Apr 12, 2019
 *      Author: oliv
 */

#include "opt.h"
#include <thread>
using namespace std;
using namespace arma;

namespace net {

void opt::grad(const Net & N,Net & G,const double & target){

	const vector<int> & layers = N.L();
	const int n_layers = layers.size();
	const vector<double (*)(const double &)> deriv=G.getFs();
//	Deal with end node if net is not empty
	if(n_layers > 0){
//		cout << N.n(n_layers-1,0) << endl;
		G.v(n_layers-1,0) = 2*deriv[n_layers-1](N.n(n_layers-1,0))*( N.v(n_layers-1,0)  -target);
//		cout <<deriv[n_layers-1](N.n(n_layers-1,0))<<endl;
	}

	for(int l = n_layers-2 ; l != -1 ; --l){
//		Coeffs of G store partial diff with respect to that coeff
		for(int start = 0 ; start != layers[l] ; ++start){
			for(int end = 0; end != layers[l+1]; ++end){
				G.c(l,start,end) = N.v(l,start)*G.v(l+1,end);
			}
		}
//		Nodes of G store partial diff with respect to that node value
		for(int start = 0 ; start != layers[l] ; ++start){
			G.v(l,start) = 0;
			for(int end = 0; end != layers[l+1]; ++end){
				G.v(l,start) += N.c(l,start,end)*deriv[l](N.n(l,start))*G.v(l+1,end);
			}
		}
	}
}

arma::mat opt::grad_descent(const arma::mat & v0, arma::mat (& grad) (const arma::mat &), const double & etha,const double & eps){
	mat v{v0};
	mat g = grad(v);
//	double old_n = norm(g);
	double eth = etha;
//	double temp = old_n;
	const int maxIt{300};
	for(int i = 0 ; i != maxIt ; ++i ){
		if(norm(g) < eps){
			cout << "numit" << i <<' ' << norm(g) <<'\n';
			return v;
		}
		else{
			v = v-eth*g;
			g = grad(v);
//			cout << "grad" << norm(g) << endl;
		}
	}
	cout << "reached max it" << norm(g) << '\n';
	return v;
}

void opt::update_partial(Net* N, const int l, const int min_node, const int max_node ) {
	const vector<int> layers = (*N).L();
	const std::vector<double (*)(const double&)> fs = (*N).getFs();

	for(int end = min_node ; end != max_node ; ++end){
		(*N).n(l,end) = 0;
		for(int start = 0 ; start != layers[l-1] ; ++start){
			(*N).n(l,end) += (*N).c(l-1,start,end)*(*N).v(l-1,start);
		}
		cout <<2<<endl;

		(*N).v(l,end) = fs[l]((*N).n(l,end)); /* function of layer */
	}

}

void opt::update(Net* N) {
	const int nThreads = (*N).getNthreads();
	vector<std::thread> trds;
	trds.reserve(nThreads);
	const vector<int> layers = (*N).L();
	const std::vector<double (*)(const double&)> fs = (*N).getFs();

	if(layers.size() >0){
		for(int end = 0 ; end != layers[0] ; ++end){
			(*N).v(0,end) = fs[0]((*N).n(0,end)); /* function of layer */
//		cout << n(0,end)<<endl;
		}
	}


	for(int l = 1 ; l != layers.size() ; l++){
		int size = layers[l]/nThreads;
		int min_node = 0;
		int max_node = 0;

		for(int i = 0 ; i != nThreads-1 ; ++i){
			min_node = max_node;
			max_node += size;
			trds[i] = thread{update_partial, N,  l,  min_node,  max_node };
		}

		min_node = max_node;
		max_node = layers[l];
		cout <<1<<endl;

		trds[nThreads-1] = thread{update_partial, N,  l,  min_node,  max_node };

		for(int i = 0 ; i != nThreads ; ++i){
			trds[i].join();
		}

	}
}

} /* namespace net */
