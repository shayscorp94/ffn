/*
 * dataFrame.h
 *
 *  Created on: Nov 11, 2018
 *      Author: oliv
 */

#ifndef DATAFRAME_H_
#define DATAFRAME_H_
#include <armadillo>
#include <iostream>
#include <vector>
#include <iomanip>      // std::setprecision

namespace vSpace {

class dataframe {
public:
	dataframe();
	dataframe(const arma::mat& data);
	dataframe(const arma::mat& data, const std::vector<std::string> & titles);
	dataframe(const int& n,const int& m, const std::string& filename,bool header = true);
	dataframe(const int& n,const std::vector<bool>& cols, const std::string& filename);

	virtual ~dataframe();

	const std::vector<std::string>& getTitles() const {
		return titles;
	}

	const arma::mat& getData() const {
		return data;
	}

	int getM() const {
		return m;
	}

	int getN() const {
		return n;
	}

	void write_csv(const std::string & filename);

    friend std::ostream& operator<<(std::ostream& os, const dataframe& D){
    	if(D.header){
    	os << "   ";
    	for(std::vector<std::string>::const_iterator it = D.getTitles().begin() ; it != D.getTitles().end() ; ++it){
    		os << *it << "   ";
    	}
    	os << '\n';
    	}

    	os << D.getData();
    	return os;
    }


private:
	int n;
	int m;
	arma::mat data;
	std::vector<std::string> titles;
	bool header;

};

} /* namespace vSpace */

#endif /* DATAFRAME_H_ */
