/*
 * DataFrame.cpp
 *
 *  Created on: Nov 11, 2018
 *      Author: oliv
 */

#include "dataframe.h"


using namespace arma;
using namespace std;
namespace vSpace {

double stod_nan(const std::string & s){
	try{
		return stod(s);
	}
	catch(std::invalid_argument){
		return 0;
	}
}

dataframe::dataframe():n(0),m(0),header(false) {
	data = mat(0,0);
	titles = std::vector<std::string>(0);

}

vSpace::dataframe::dataframe(const arma::mat& data):
	data(data),n(data.n_rows),m(data.n_cols),header(false){
	titles = std::vector<std::string>(0);
}

dataframe::dataframe(const arma::mat& data,const std::vector<std::string>& titles):
		data(data),n(data.n_rows),m(data.n_cols),titles(titles),header(true){}

dataframe::dataframe(const int& n, const int& m, const std::string& filename,bool header):
		n(n),m(m),header(header) {
	data = mat(n,m);
	titles = std::vector<std::string>(m);

	string temp{};
	ifstream ip(filename);

//	extract the titles
	if(header){
	for(int j = 0 ; j != m-1 ; ++j){
		getline(ip,temp,',');
		titles[j] = temp;
	}
	getline(ip,temp,'\n');
	titles[m-1] = temp;
	}
//	extract the values (double)

	for(int i = 0 ; i != n ; ++i){
		for(int j = 0 ; j != m-1 ; ++j){
			getline(ip,temp,',');
//			cout << j << ' '<< temp << endl;
			data(i,j) = stod_nan(temp);
		}
		getline(ip,temp,'\n');
//		cout << temp << endl;
		data(i,m-1) = stod_nan(temp);
	}

	ip.close();
}

dataframe::dataframe(const int& n, const std::vector<bool>& cols,
		const std::string& filename): n(n),m(0),header(true) {

//	initialisation of m :
	for(vector<bool>::const_iterator it = cols.begin() ; it != cols.end() ; ++it){
	if(*it){ ++m;	}
	}
	data = mat(n,m);
	titles = std::vector<std::string>(m);

	string temp{};
	ifstream ip(filename);

//	extract the titles
	int pos{0};
	for(int j = 0 ; j != cols.size()-1 ; ++j){
		getline(ip,temp,',');
		if(cols[j]){titles[pos] = temp;++pos;}
	}
	getline(ip,temp,'\n');
	if(cols[cols.size()-1]){titles[m-1] = temp;}

//	extract the values (double)
	for(int i = 0 ; i != n ; ++i){
		pos = 0;
		for(int j = 0 ; j != cols.size()-1 ; ++j){
			getline(ip,temp,',');
			if(cols[j]){data(i,pos) = stod(temp);++pos;}
		}
		getline(ip,temp,'\n');
		if(cols[cols.size()-1]){data(i,pos) = stod(temp);}
	}

	ip.close();

}
void dataframe::write_csv(const std::string& filename) {
	  ofstream myfile (filename);
	  if (myfile.is_open())
	  {
		if(header){
		for(int j = 0 ; j != m-1 ; ++j){
			myfile << titles[j]<<',';
		}
		myfile << titles[m-1] << '\n';
		}
	    for(int i = 0 ; i != n ; ++i){
	    	for(int j = 0 ; j != m-1 ; ++j){
	    		myfile <<std::setprecision(20)<< data(i,j) <<',';
	    	}
	    	myfile <<std::setprecision(40)<< data(i,m-1)<<'\n';
	    }
	    myfile.close();
	  }
	  else cout << "Unable to open file";
}


dataframe::~dataframe() {
	// TODO Auto-generated destructor stub
}

} /* namespace vSpace */


