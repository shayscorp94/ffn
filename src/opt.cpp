/*
 * opt.cpp
 *
 *  Created on: Apr 12, 2019
 *      Author: oliv
 */

#include "opt.h"
#include <thread>
#include <random>
#include <algorithm> // std::random_shuffle



using namespace std;
using namespace arma;

namespace net {

void opt::grad(const Net & N,const int sample,Net & G,const double & target){

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

arma::vec opt::grad_descent(Net * N, const double & etha,const double & eps, bool linSearch){
	double eth = etha;
	const int maxIt{100};
//	vector<double (*)(const double &)> ds{opt::One,opt::DLrelu,opt::DLrelu,opt::DLrelu};

//	For threads
	const int nsamples =(*N).getNsamples();
	int n_threads = (*N).getNthreads();
	n_threads = (n_threads>nsamples) ?nsamples:n_threads; /* prevent from having more threads than samples */
    /* the two next are needed to avoid datarace between threads and will be passed to opt::gradient */
	vector<Net> Gs(n_threads, Net( (*N).L(),(*N).getDs(),(*N).getDs(),1 ) ); /* one Net per thread, it is used by grad to caluclate the gradient, use the derivatives as activation functions (see implementation of grad)*/
	mat thread_grads((*N).getNcoeffs(),n_threads,fill::zeros); /* one column per thread */

	vec g((*N).getNcoeffs(),fill::zeros); /* stores the gradient (sum of thread_grads divided by nsamples) */

//	time measurement
	auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed;

//	for linsearch
	double lmin = 0;
	double min = -1 ;
	double lin = 0.1;
	double error = 0;
	const double minlr = 1e-9;

//	initial gradient
	gradient(N,&Gs,&g,&thread_grads);


	for(int i = 0 ; i != maxIt ; ++i ){
		if(norm(g) < eps){ /* stopping condition */
			finish = std::chrono::high_resolution_clock::now();
			elapsed = finish - start;
			cout << "numit" << i <<' '<< norm(g) <<" time"<<elapsed.count()<<'\n';
			return (*N).get_coeffs();
		}
		else{
			if(linSearch){
			lmin = 0;
			min = -1 ;
			lin = 0.1;
			for(int j = 0 ; j != 3 ; ++j){ /* try 0.1* eth, eth and 10*eth and choose the best one */
				(*N).get_coeffs() -= eth*lin*g;
				error = err(N); /* compute the error using parallelization */
				(*N).get_coeffs() += eth*lin*g; /* reset the coeffs */


				if( min == -1 or min > error){ /* min == -1 iff j = 0 */
					min = error;
					lmin = lin;
				}
				lin *= 10;
			}
			eth = lmin*eth;
//			If eth becomes too small we keep and and stop doing linear searches.
			linSearch = (eth > minlr)?true:false;
			}

			(*N).get_coeffs() -= eth*g; /* use the "optimal" learning rate*/
			gradient(N,&Gs,&g,&thread_grads);

		}
	}
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << "reached max it" << norm(g) <<" time"<<elapsed.count()<<'\n';
	return (*N).get_coeffs();
}

void opt::partial_grad(arma::mat * thread_grads, Net * N,std::vector<Net> * Gs,const int start,const int end,const int thread ){
	for(int sample = start; sample != end ; ++sample){
		(*N).update(sample);
		grad(*N,sample,(*Gs)[thread],(*N).getTarget(sample));
		(*thread_grads).col(thread) += (*Gs)[thread].get_coeffs();
	}
}

void opt::gradient( Net * N,std::vector<Net> * Gs,arma::vec * res_grad,arma::mat * thread_grads){

	const int nsamples =(*N).getNsamples();
	int n_threads = (*thread_grads).n_cols;
	vector<thread> trds(n_threads); /* vetor that stores the threads*/
	(*thread_grads).fill(0.); /* empty the gradient matrix */
	int start = 0;
	int end = 0;
	int size = nsamples/n_threads; /* size of the chunk given to each thread, last thread might have less */

		for(int t = 0 ; t != n_threads-1 ; ++t){
			start = end;
			end += size;
			trds[t] = thread{partial_grad,thread_grads,N,Gs,start,end,t}; /* call partial grad*/
		}
		/* deals with case where nsamples not a multiple of nthreads -> end = nsamples */
			start = end;
			end  = nsamples;
			trds[n_threads-1] = thread{partial_grad,thread_grads,N,Gs,start,end,n_threads-1};

		for(int t= 0 ; t != n_threads ; ++t){
			trds[t].join();
		}
	*res_grad = sum(*thread_grads,1)/nsamples; /* total gradient */
};

void opt::partial_err(double * res, Net * N,const int start,const int end,const int thread ){
	const int nlayers = (*N).L().size();
	for(int sample = start; sample != end ; ++sample){
		(*N).update(sample);
		*res += pow ( (*N).n(sample,nlayers-1,0) - (*N).getTarget(sample), 2  );
	}
}

double opt::err(Net * N){ /* similar as gradient function */
//	cout << "opt::gradient"<<endl;
	const int nsamples =(*N).getNsamples();
	int n_threads = (*N).getNthreads();
	n_threads = (n_threads>nsamples) ?nsamples:n_threads; /* prevent from having more threads than samples */
	vector<thread> trds(n_threads);
	int start = 0;
	int end = 0;
	int size = nsamples/n_threads;
	vec res(n_threads,fill::zeros);

		for(int t = 0 ; t != n_threads-1 ; ++t){
			start = end;
			end += size;
			trds[t] = thread{partial_err,&(res(t)),N,start,end,t};
		}
			start = end;
			end  = nsamples;
			trds[n_threads-1] = thread{partial_err,&(res(n_threads-1)),N,start,end,n_threads-1};

		for(int t= 0 ; t != n_threads ; ++t){
			trds[t].join();
		}
	return sum(res)/nsamples;
};

void opt::result( Net* N) {
	const int nsamples =(*N).getNsamples();
	const int nlayers =(*N).L().size();
		cout <<"prediction      target\n";
	for(int sample = 0 ; sample != nsamples ; ++sample){
		cout << (*N).n(sample,nlayers-1,0) << "         " <<(*N).getTarget(sample)<<'\n';
	}
		cout <<"\nerror "<< err(N);
}

arma::mat opt::stochastic_descent(Net * N, const double & etha,const double & eps,const int batchSize, bool linSearch,const int & nepochs,const double & minlr){
	/* one epochs corresponds to nsapmles/nbatches step. On step is : compute gradient of the batch , lin search , descend along the gradient */
	double eth = etha;
	const int maxIt{nepochs};

	const int nsamples = (*N).getNsamples();

//	for threads
	int n_threads = (*N).getNthreads();
	n_threads = (n_threads>batchSize) ?batchSize:n_threads; /* prevent from having more threads than samples */

	vector<Net> Gs(n_threads, Net( (*N).L(),(*N).getDs(),(*N).getDs(),1 ) );

	mat thread_grads((*N).getNcoeffs(),n_threads,fill::zeros);

	vec g((*N).getNcoeffs(),fill::zeros);


//	time measurement
	auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed;

//	for linsearch
	double lmin = 0;
	double min = -1 ;
	double lin = 0.1;
	double error = 0;
//	const double minlr = 1e-9;



	int batchStart = 0;
	int batchEnd = 0;
//	Stats
	mat stats(maxIt,2,fill::zeros);

//	permutations
	vec sigma(nsamples);
	for(int i = 0; i != nsamples ; ++i){
		sigma(i) = i;
	}
//  before each epoch, permute the samples : sigma will be passed to gradient_st
//	actually we do no permute the samples we just use sigma wich indicates which indexes to use
	for(int i = 0; i != maxIt ; ++i){
		sigma = shuffle(sigma);

		batchStart = 0;
		batchEnd = 0;
		for(int j = 0 ; j != nsamples/batchSize-1 ; ++j){
			batchStart = batchEnd;
			batchEnd += batchSize;

			gradient_st(N,&Gs,&g,&thread_grads,&sigma,batchStart,batchEnd);

			if(linSearch){
			lmin = 0;
			min = -1 ;
			lin = 0.1;
			for(int j = 0 ; j != 3 ; ++j){
				(*N).get_coeffs() -= eth*lin*g;
				error = err_st(N,&sigma,batchStart,batchEnd);
				(*N).get_coeffs() += eth*lin*g;


				if( min == -1 or min > error){
					min = error;
					lmin = lin;
				}
				lin *= 10;
			}
			eth = lmin*eth;
	//			If eth becomes too small we keep and and stop doing linear searches.
			linSearch = (eth > minlr)?true:false;
			}
//			cout << i<<endl;

			(*N).get_coeffs() -= eth*g;

		}
		batchStart = batchEnd;
		batchEnd = nsamples;

		gradient_st(N,&Gs,&g,&thread_grads,&sigma,batchStart,batchEnd);
//		cout << norm(g)<<'\n';

		if(norm(g) < eps){
				finish = std::chrono::high_resolution_clock::now();
				elapsed = finish - start;
				vec end_stats(3);
				end_stats(0) = elapsed.count();
				end_stats(1) = norm(g);
				end_stats(2) = err(N);
				cout << "numit" << i <<" time"<<end_stats(0)<<"norm"<< end_stats(1)<<"err"<<end_stats(2)<<'\n';
				return stats;
			}
		if(linSearch){
		lmin = 0;
		min = -1 ;
		lin = 0.1;
		for(int j = 0 ; j != 3 ; ++j){
			(*N).get_coeffs() -= eth*lin*g;
			error = err_st(N,&sigma,batchStart,batchEnd);
			(*N).get_coeffs() += eth*lin*g;


			if( min == -1 or min > error){
				min = error;
				lmin = lin;
			}
			lin *= 10;
		}
		eth = lmin*eth;

//			If eth becomes too small we keep and and stop doing linear searches.
		linSearch = (eth > minlr)?true:false;
		}
		(*N).get_coeffs() -= eth*g;
//		stats
		stats(i,0) =norm(g);
		stats(i,1) =err(N);

//		cout << norm(g)<<'\n';

	}
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	vec end_stats(3);
	end_stats(0) = elapsed.count();
	end_stats(1) = norm(g);
	end_stats(2) = err(N);

	cout << "maxEpochs" << maxIt <<" time"<<end_stats(0)<<"norm"<< end_stats(1)<<"err"<<end_stats(2)<<'\n';
	return stats;
}

void opt::gradient_st( Net * N,std::vector<Net> * Gs,arma::vec * res_grad,arma::mat * thread_grads,const vec * sigma,const int batchStart,const int batchEnd){
//	cout << "opt::gradient"<<endl;
	const int nsamples =batchEnd-batchStart; /* nsamples is not the total number of samples but just the number of samples in the batch, can be different of batchSize for the last batch of the epoch */
	int n_threads = (*thread_grads).n_cols;
	n_threads = (n_threads>nsamples) ?nsamples:n_threads; /* prevent from having more threads than samples */

	(*thread_grads).fill(0.);

	vector<thread> trds(n_threads);
	int start = batchStart;
	int end = batchStart;
	int size = nsamples/n_threads;

		for(int t = 0 ; t != n_threads-1 ; ++t){
			start = end;
			end += size;
			trds[t] = thread{partial_grad_st,thread_grads,N,Gs,sigma,start,end,t};
		}
			start = end;
			end  = batchEnd;

			trds[n_threads-1] = thread{partial_grad_st,thread_grads,N,Gs,sigma,start,end,n_threads-1};

		for(int t= 0 ; t != n_threads ; ++t){
			trds[t].join();
		}
	*res_grad = sum(*thread_grads,1)/nsamples; /* One column per thread*/
};


void opt::partial_grad_st(arma::mat * thread_grads, Net * N,std::vector<Net> * Gs,const vec * sigma,const int start,const int end,const int thread ){
//	cout << "opt::partial_grad"<< start << " "<< end << " "<< thread<<endl;

	for(int sample = start; sample != end ; ++sample){
//		Use sigma to use the permuted samples
		(*N).update((*sigma)(sample));
		grad(*N,(*sigma)(sample),(*Gs)[thread],(*N).getTarget((*sigma)(sample)));
		(*thread_grads).col(thread) += (*Gs)[thread].get_coeffs();
	}
}

double opt::err_st(Net * N,const arma::vec * sigma,const int batchStart,const int batchEnd){
//	cout << "opt::gradient"<<endl;
	const int nsamples =batchEnd-batchStart;
	int n_threads = (*N).getNthreads();
	n_threads = (n_threads>nsamples) ?nsamples:n_threads; /* prevent from having more threads than samples */
	vector<thread> trds(n_threads);
	int start = batchStart;
	int end = batchStart;
	int size = nsamples/n_threads;
	vec res(n_threads,fill::zeros);

		for(int t = 0 ; t != n_threads-1 ; ++t){
			start = end;
			end += size;
			trds[t] = thread{partial_err_st,&(res(t)),N,sigma,start,end,t};
		}
			start = end;
			end  =batchEnd;
			trds[n_threads-1] = thread{partial_err_st,&(res(n_threads-1)),N,sigma,start,end,n_threads-1};

		for(int t= 0 ; t != n_threads ; ++t){
			trds[t].join();
		}
	return mean(res);
};

void opt::partial_err_st(double * res, Net * N,const arma::vec * sigma,const int start,const int end,const int thread ){
	const int nlayers = (*N).L().size();
	for(int sample = start; sample != end ; ++sample){
		(*N).update((*sigma)(sample));
		*res += pow ( (*N).n((*sigma)(sample),nlayers-1,0) - (*N).getTarget((*sigma)(sample)), 2  );
	}
}



//
//double opt::err_st(Net * N,mt19937 * gen,const int batchSize){
////	cout << "opt::gradient"<<endl;
//	int n_threads = (*N).getNthreads();
//	n_threads = (n_threads>batchSize) ?batchSize:n_threads; /* prevent from having more threads than samples */
//	vector<thread> trds(n_threads);
//	int start = 0;
//	int end = 0;
//	int size = batchSize/n_threads;
//	double res=0;
//
//		for(int t = 0 ; t != n_threads-1 ; ++t){
//			start = end;
//			end += size;
//			trds[t] = thread{partial_err_st,&res,N,gen,start,end,t};
//		}
//			start = end;
//			end  = batchSize;
//			trds[n_threads-1] = thread{partial_err_st,&res,N,gen,start,end,n_threads-1};
//
//		for(int t= 0 ; t != n_threads ; ++t){
//			trds[t].join();
//		}
//	return res /=batchSize;
//};





} /* namespace net */
