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

void opt::grad(const Net & N,const int sample,Net & G,const double & target){
//	cout << "opt::grad" << endl;


	const vector<int> & layers = N.L();
	const int n_layers = layers.size();
	const vector<double (*)(const double &)> deriv=G.getFs();
//	Deal with end node if net is not empty
	if(n_layers > 0){
		G.v(0,n_layers-1,0) = 2*deriv[n_layers-1](N.n(sample,n_layers-1,0))*( N.v(sample,n_layers-1,0)  -target);
	}

	for(int l = n_layers-2 ; l != -1 ; --l){
//		Coeffs of G store partial diff with respect to that coeff
		for(int start = 0 ; start != layers[l] ; ++start){
			for(int end = 0; end != layers[l+1]; ++end){
				G.c(l,start,end) = N.v(sample,l,start)*G.v(0,l+1,end);
			}
		}
//		Nodes of G store partial diff with respect to that node value
		for(int start = 0 ; start != layers[l] ; ++start){
			G.v(0,l,start) = 0;
			for(int end = 0; end != layers[l+1]; ++end){
				G.v(0,l,start) += N.c(l,start,end)*deriv[l](N.n(sample,l,start))*G.v(0,l+1,end);
			}
		}
	}
}

arma::vec opt::grad_descent(Net * N, const double & etha,const double & eps){
//	cout << "opt::grad_descent"<<endl;
	double eth = etha;
	const int maxIt{1000};
	vector<double (*)(const double &)> ds{opt::One,opt::DLrelu,opt::DLrelu,opt::DLrelu};
	vector<Net> Gs((*N).getNthreads(), Net((*N).L(),ds,1) );


	mat g = gradient(N,&Gs);
	cout << norm(g)<<"\n";


	for(int i = 0 ; i != maxIt ; ++i ){
		if(norm(g) < eps){
			cout << "numit" << i <<' ' << norm(g) <<'\n';
			return (*N).get_coeffs();
		}
		else{
			(*N).get_coeffs() -= eth*g;
			g = gradient(N,&Gs);

		}
	}
	cout << "reached max it" << norm(g) << '\n';
	return (*N).get_coeffs();
}

void opt::partial_grad(arma::vec * res, Net * N,std::vector<Net> * Gs,const int start,const int end,const int thread ){
//	cout << "opt::partial_grad"<< start << " "<< end << " "<< thread<<endl;

	for(int sample = start; sample != end ; ++sample){
		(*N).update(sample);

		grad(*N,sample,(*Gs)[thread],(*N).getTarget(sample));


		*res += (*Gs)[thread].get_coeffs();
	}
}

arma::mat opt::gradient( Net * N,std::vector<Net> * Gs){
//	cout << "opt::gradient"<<endl;
	vec res_grad((*N).getNcoeffs(),fill::zeros);
	const int nsamples =(*N).getNsamples();
	int n_threads = (*N).getNthreads();
	n_threads = (n_threads>nsamples) ?nsamples:n_threads; /* prevent from having more threads than samples */
	vector<thread> trds(n_threads);
	int start = 0;
	int end = 0;
	int size = nsamples/n_threads;

		for(int t = 0 ; t != n_threads-1 ; ++t){
			start = end;
			end += size;
			trds[t] = thread{partial_grad,&res_grad,N,Gs,start,end,t};
		}
			start = end;
			end  = nsamples;
			trds[n_threads-1] = thread{partial_grad,&res_grad,N,Gs,start,end,n_threads-1};

		for(int t= 0 ; t != n_threads ; ++t){
			trds[t].join();
		}
	return res_grad/nsamples;
};

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
