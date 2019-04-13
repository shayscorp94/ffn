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

void opt::grad(const Net & N,const int sample,Net & G,const int thread,const double & target){

	const vector<int> & layers = N.L();
	const int n_layers = layers.size();
	const vector<double (*)(const double &)> deriv=G.getFs();
//	Deal with end node if net is not empty
	if(n_layers > 0){
//		cout << N.n(n_layers-1,0) << endl;
		G.v(thread,n_layers-1,0) = 2*deriv[n_layers-1](N.n(sample,n_layers-1,0))*( N.v(sample,n_layers-1,0)  -target);
//		cout <<deriv[n_layers-1](N.n(n_layers-1,0))<<endl;
	}

	for(int l = n_layers-2 ; l != -1 ; --l){
//		Coeffs of G store partial diff with respect to that coeff
		for(int start = 0 ; start != layers[l] ; ++start){
			for(int end = 0; end != layers[l+1]; ++end){
				G.c(l,start,end) = N.v(sample,l,start)*G.v(thread,l+1,end);
			}
		}
//		Nodes of G store partial diff with respect to that node value
		for(int start = 0 ; start != layers[l] ; ++start){
			G.v(thread,l,start) = 0;
			for(int end = 0; end != layers[l+1]; ++end){
				G.v(thread,l,start) += N.c(l,start,end)*deriv[l](N.n(sample,l,start))*G.v(thread,l+1,end);
			}
		}
	}
}

void opt::grad_descent(Net & N, arma::mat (& grad) (const Net *,vector<Net> *), const double & etha,const double & eps){
	double eth = etha;
	const int maxIt{1000};
	const int nthreads;
	vector<Net> Gs(N.getNthreads(), Net(N.L(),N.getFs(),1) );


	mat g = grad(N,Gs);

	for(int i = 0 ; i != maxIt ; ++i ){
		if(norm(g) < eps){
			cout << "numit" << i <<' ' << norm(g) <<'\n';
//			return v;
		}
		else{
			N.get_coeffs() -= eth*g;
			g = grad(N,Gs);

		}
	}
	cout << "reached max it" << norm(g) << '\n';
//	return v;
}

inline void partial_grad(arma::vec * res,const Net * N,Net * G,const int start,const int end,const int thread ){
	for(int sample = start; sample != end ; ++sample){
		(*N).update(sample);
		opt::grad(*N,sample,*G,thread,(*N).target(sample));
		*res += (*G).get_coeffs();
	}
}

//void opt::update_partial(Net* N, const int l, const int min_node, const int max_node ) {
//	const vector<int> layers = (*N).L();
//	const std::vector<double (*)(const double&)> fs = (*N).getFs();
//
//	for(int end = min_node ; end != max_node ; ++end){
//		(*N).n(l,end) = 0;
//		for(int start = 0 ; start != layers[l-1] ; ++start){
//			(*N).n(l,end) += (*N).c(l-1,start,end)*(*N).v(l-1,start);
//		}
////		cout <<end<<endl;
//
//		(*N).v(l,end) = fs[l]((*N).n(l,end)); /* function of layer */
//	}
//
//}
//
//void opt::update(Net* N) {
//	const int nThreads = (*N).getNthreads();
////	cout << nThreads<<'\n';
//	const vector<int> layers = (*N).L();
//	const std::vector<double (*)(const double&)> fs = (*N).getFs();
//
//	if(layers.size() >0){
//		for(int end = 0 ; end != layers[0] ; ++end){
//			(*N).v(0,end) = fs[0]((*N).n(0,end)); /* function of layer */
////		cout << n(0,end)<<endl;
//		}
//	}
//
//
//	for(int l = 1 ; l != layers.size()-1 ; l++){
//		vector<std::thread> trds;
//		trds.reserve(nThreads);
//
//
//		int size = layers[l]/nThreads;
//		int min_node = 0;
//		int max_node = 0;
//
//		for(int i = 0 ; i != nThreads-1 ; ++i){
//			min_node = max_node;
//			max_node += size;
//			trds.push_back(thread{update_partial, N,  l,  min_node,  max_node });
////			cout << "yo";
////			update_partial(N,  l,  min_node,  max_node );
//
//		}
//
//		min_node = max_node;
//		max_node = layers[l];
////		cout <<1<<endl;
////		cout << l <<' '<< min_node<<' ' << max_node<<' ' <<endl;
//
//		trds.push_back(thread{update_partial, N,  l,  min_node,  max_node });
////		update_partial( N,  l,  min_node,  max_node );
//		for(int i = 0 ; i != nThreads ; ++i){
////			cout << "joinable"<< trds[i].joinable() <<' ';
//			trds[i].join();
//		}
//
//	}
//}

} /* namespace net */
