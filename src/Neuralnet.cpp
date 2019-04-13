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


static const int nassets{487};
static const int nlines{756};
static const double end_train{11};
static dataframe Data{756,nassets,"cleanIndex.csv"};
static mat Train = Data.getData().rows(0,end_train+0);
static const int batchSize{1};


vector<double (*)(const double &)> fs{opt::I,opt::Lrelu,opt::Lrelu,opt::Lrelu};
vector<double (*)(const double &)> ds{opt::One,opt::DLrelu,opt::DLrelu,opt::DLrelu};
vector<Net> nets;
vector<int> layers{487,250,125,1};
vector<Net> grads(2,Net(layers,ds));

Net G = Net(vector<int>{487,250,125,1},ds); /* we do not care of the target we just want the structure*/
vec res_grad = vec(G.get_coeffs().n_rows,fill::zeros);



inline void multi_grad(arma::vec * res,Net * G,const arma::mat * v,const int start,const int end ){
	for(int d = start; d != end ; ++d){
		nets[d].get_coeffs() = *v;
		nets[d].update();
		opt::grad(nets[d],*G,Train(d+10,0));
		*res += (*G).get_coeffs();
	}
}

inline arma::mat g(const arma::mat * v){
res_grad.fill(0.);
const int n_dates =end_train+1-10;
const int n_threads = 2;
vector<thread> trds(n_threads);
int start = 0;
int end = 0;
int size = n_dates/n_threads;

	for(int t = 0 ; t != n_threads-1 ; ++t){
		start = end;
		end += size;
		trds[t] = thread{multi_grad,&res_grad,&grads[t],v,start,end};
	}
		start = end;
		end  = n_dates;
		trds[n_threads-1] = thread{multi_grad,&res_grad,&grads[n_threads-1],v,start,end};

	for(int t= 0 ; t != n_threads ; ++t){
		trds[t].join();
	}
return res_grad/(end_train+1-10);
};

inline void multi_err(double * err,const arma::mat * v,const int start,const int end ){
	for(int d = start; d != end ; ++d){
		nets[d].get_coeffs() = *v;
		nets[d].update();
		opt::grad(nets[d],*G,Train(d+10,0));
		*res += (*G).get_coeffs();
	}
}



//
//
//arma::mat acc_descent(const arma::mat& v0 , std::function<arma::mat(const arma::mat &)> & grad,const double & eth,const double & eps){
//	mat x{v0};
//	mat y{v0};
//	mat y_old{v0};
//	mat g = grad(v0);
//	double s{0};
//	const int maxIt{100};
//	for(int i = 0 ; i != maxIt ; ++i ){
//		if(norm(g) < eps){
//			cout << "numit" << i <<' ' << norm(g) <<'\n';
//			return x;
//		}
//		else{
//			s = ((double)i)/(i+3.);
//			y = x - eth*g;
//			x = x*(1+s)-eth*g*(1+s)-s*y_old;
//			g = grad(x);
//			y_old = y;
////			cout << norm(g)<<endl;
//		}
//	}
//	cout << "reached max it" << norm(g) << '\n';
//	return x;
//
//}

arma::mat stochastic_descent(const arma::mat& v0, std::function<arma::mat(const arma::mat&,mt19937 &)> & grad,const double & etha,const double & eps){
	mat v{v0};
	const int maxIt{400};
	mt19937 gen;
	mt19937 genB;
	uniform_int_distribution<int> Dist(0,::end_train-10);
	mat g = grad(v,gen);
	double eth = etha;
	double lmin = 0;
	double min = -1 ;
	double lin = 0.1;
	double err = 0;
	int d{0};
	const int n_layers = 4;

	for(int i = 0 ; i != maxIt ; ++i ){
		if(norm(g) < eps){
			cout << "numit" << i <<' ' << norm(g) <<'\n';
			return v;
		}
		else{
			lmin = 0;
			min = -1 ;
			lin = 0.1;
			for(int j = 0 ; j != 3 ; ++j){
				err = 0;

				for(int k = 0 ; k != batchSize ; ++k){
				d = Dist(genB);
				nets[d].get_coeffs() = v-eth*lin*g;
				nets[d].update();
				err += pow(nets[d].v(n_layers-1,0)-Train(10,0),2);
				}

				if( min == -1 or min > err){
//					cout << pow(N.v(4-1,0)-Train(10,0),2)<< endl;
					min = err;
					lmin = lin;
				}
				lin *= 10;
			}
			eth = lmin*eth;
//			cout << lmin <<endl;

			v = v-eth*g;


			g = grad(v,gen);
			if(i % 100000 == 0){
			cout << norm(g) << '\n';
			}
		}
	}
	cout << "reached max it" << norm(g) << '\n';
	return v;
}

int main(){

//	auto start = std::chrono::high_resolution_clock::now();


	nets.reserve(::end_train+1-10);
	for(int d = 0 ; d != ::end_train+1-10 ; ++d){
	//		For each available date, we calculate a gradient and then we average
		nets.push_back(Net(layers,fs,2));
		nets[d].n(0,0) = 0;
		for(int s = 1 /* do not use the current index price*/ ; s != nassets ; ++s){
			nets[d].n(0,s) = Train(d,s);
		}
		nets[d].update();
//		opt::update(&nets[d]);
	}
//	auto finish = std::chrono::high_resolution_clock::now();
//	std::chrono::duration<double> elapsed = finish - start;
//
//	cout << elapsed.count();



//	std::function<arma::mat(const arma::mat &, mt19937 & )> g_st = [&](const arma::mat & v, mt19937 & g){
//	uniform_int_distribution<int> Dist(0,::end_train-10);
////	const int batchSize{2};
//	res_grad.fill(0.);
//	int d{0};
//	for(int i = 0 ; i!= ::batchSize ; ++i){
//		d = Dist(g);
////		cout << d<<endl;
//		::nets[d].get_coeffs() = v;
//		nets[d].update();
//		opt::grad(nets[d],G,::Train(d+10,0));
//		res_grad += G.get_coeffs();
//	}
////	cout << ::nets[0].v(4,0)<<endl;
//	return res_grad/(::batchSize);
//	};




//	dataframe dv0(nets[0].get_coeffs().n_rows,1,"v5.csv",false);
//	vec v0 = dv0.getData();
//	vec v0{nets[0].get_coeffs().n_rows,fill::randn};
	vec v0 = nets[0].get_coeffs();


	auto start = std::chrono::high_resolution_clock::now();
	vec vinf = opt::grad_descent(v0,g,1e-8,0.00000001);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;

	dataframe initVec{vinf};
	initVec.write_csv("v6.csv");

	cout << norm( v0 - vinf)/v0.n_rows <<endl;
	cout << "Prediciton   Real Val"<<'\n';
	double err = 0;
	for(int d = 0 ; d != end_train+1-10 ; ++d){
//		nets[d].get_coeffs() = v0;
//		nets[d].update();
		err += pow(nets[d].v(3,0) - Train(d+10,0),2);
		cout << nets[d].v(3,0) << "           "<<Train(d+10,0)<<'\n';
	}
	cout << err/( end_train+1-10);
	cout << "time" << elapsed.count();



	return 0;
}


