/*
 * Net.cpp
 *
 *  Created on: Apr 12, 2019
 *      Author: oliv
 */

#include "Net.h"

namespace net {

#include "Net.h"
#include "math.h"
#include <random>
using namespace std;
using namespace arma;


Net::Net(const std::vector<int> & layers, std::vector<double (*)(const double&)> fs,const int & nsamples,const int & nthreads)
:layers(layers),fs(fs),nsamples(nsamples),nthreads(nthreads)
{
	int n_coeffs{0};
	int n_nodes{0};
	int temp{1};

	for( vector<int>::const_iterator it = layers.begin(); it != layers.end() ; ++it){
		if(it == layers.begin()){
			temp = *it;
			n_nodes += temp;
		}
		else{
			n_coeffs += temp * *it;
			temp = *it;
			n_nodes += temp;
		}
	}

	coeffs = vec(n_coeffs, fill::zeros);
	mt19937 g;
	std::normal_distribution<double> d;

	temp = 0;
	for (vector<int>::const_iterator it = layers.begin(); it != layers.end(); ++it) {
		if (it == layers.begin()) {
		}
		else {
			for(int i = temp ; i != temp + *it * *(it - 1) ;++i ){
			coeffs(i) = d(g)*sqrt(2./ *(it - 1));
			}
			temp += *it * *(it - 1);
		}
	}

	nodevals = vector<arma::vec>(nsamples,vec(n_nodes,fill::zeros));
	nodes = vector<arma::vec>(nsamples,vec(n_nodes,fill::zeros));
	targets = vec(nsamples,fill::zeros);
}

double& Net::c(const int& layer_num, const int& start, const int& end) {
	int cumsum{0};
	for(int i = 1; i !=layer_num+1;++i){
		cumsum += layers[i]*layers[i-1];
	}
	return coeffs(cumsum+start*layers[layer_num+1]+end);
}

double Net::c(const int& layer_num, const int& start, const int& end) const{
	int cumsum{0};
	for(int i = 1; i !=layer_num+1;++i){
		cumsum += layers[i]*layers[i-1];
	}
	return coeffs(cumsum+start*layers[layer_num+1]+end);
}

double& Net::v(const int & sample,const int& layer_num, const int& i) {
	int cumsum{0};
	for(int i = 0; i !=layer_num;++i){
		cumsum += layers[i];
	}
	return nodevals[sample](cumsum+i);
}

double Net::v(const int & sample,const int& layer_num, const int& i) const {
	int cumsum{0};
	for(int i = 0; i !=layer_num;++i){
		cumsum += layers[i];
	}
	return nodevals[sample](cumsum+i);
}

double& Net::n(const int & sample,const int& layer_num, const int& i) {
	int cumsum{0};
	for(int i = 0; i !=layer_num;++i){
		cumsum += layers[i];
	}
	return nodes[sample](cumsum+i);
}

double Net::n(const int & sample,const int& layer_num, const int& i) const {
	int cumsum{0};
	for(int i = 0; i !=layer_num;++i){
		cumsum += layers[i];
	}
	return nodes[sample](cumsum+i);
}

arma::vec& Net::get_coeffs() {
	return coeffs;
}

void Net::print(const int & sample) {
	int temp{0};
	cout << "VALUES"<<endl;
	for( vector<int>::const_iterator it = layers.begin(); it != layers.end() ; ++it){
			cout << nodevals[sample].rows(temp,temp+*it-1)<<endl;
			temp += *it;
	}
	temp = 0;
	cout << "COEFFICIENTS"<<endl;
	for( vector<int>::const_iterator it = layers.begin(); it != layers.end() ; ++it){
		if(it == layers.begin()){
		}
		else{
			cout << coeffs.rows(temp,temp+ *it * *(it-1) -1)<<endl;
			temp += *it * *(it-1);
		}
	}
}

void Net::update(const int & sample) {
	if(layers.size() >0){
		for(int end = 0 ; end != layers[0] ; ++end){
		v(sample,0,end) = fs[0](n(sample,0,end)); /* function of layer */
		}
	}
	for(int l = 1 ; l != layers.size() ; l++){
		for(int end = 0 ; end != layers[l] ; ++end){
			n(sample,l,end) = 0;
			for(int start = 0 ; start != layers[l-1] ; ++start){
				n(sample,l,end) += c(l-1,start,end)*v(sample,l-1,start);
			}
			v(sample,l,end) = fs[l](n(sample,l,end)); /* function of layer */
		}
	}
}

Net::~Net() {
	// TODO Auto-generated destructor stub
}


} /* namespace net */
