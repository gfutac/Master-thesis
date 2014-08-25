#include <iostream>

#include "pca.h"

using namespace std;
using namespace cv;

pca::pca(const Mat &data, int label, int vectors) : data(data), label(label), vectors(vectors) {
	if (vectors == 0){
		vectors = data.cols - 1;
	}
	calculate();
}

int pca::getLabel() { return this->label; };

void pca::calcMean(){
	float *m = new float[data.rows];
	for (int i = 0; i < data.rows; ++i){
		m[i] = sum(data.row(i)).val[0] / data.cols;
	}

	Mat tmp(data.rows, 1, CV_32F, m);
	this->mean = tmp.clone();
	
	delete [] m;	
}

void pca::calculate(){
	// calculate mean
	calcMean();
	// mean sample
	Mat tmp_mean = repeat(mean, data.rows/mean.rows, data.cols/mean.cols);
	data.convertTo( data, CV_32F);
	// meaned samples
	subtract(data, tmp_mean, data); 
	// calculate cov matrix
	Mat covar = data.t() * data; 
	// calculate eigen values and vectors
	Mat evals, evecs;
	eigen(covar, evals, evecs); 
	// reduce dimensionality, take only vectors with highest eigenvalues
	evecs = evecs.rowRange(0, vectors).clone(); 
	// eigen{faces, palms...}
	eigens = data * evecs.t();

	// normalize eigen{faces, palms...}
	for (int i = 0; i < eigens.cols; ++i){
		normalize(eigens.col(i), eigens.col(i));
	}
	
	// project!!
	projections = eigens.t() * data;
}


