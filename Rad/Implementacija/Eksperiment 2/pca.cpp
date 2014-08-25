#include <iostream>
#include <cmath>
#include <ctime>

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
	
	delete m;	
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

// cpu version of project
Mat pca::project(const Mat &sample){
	Mat s;
	if (sample.type() != CV_32F)
		sample.convertTo(s, CV_32F);

	// sample - mean
	subtract(s, mean, s);
	// return projection
	return eigens.t() * s;
}

// cpu version of min distance
double pca::minDistance(const Mat &sample){
	double x = clock();		
	Mat q = project(sample);

	double mindist = DBL_MAX;
	for (int i = 0; i < projections.cols; ++i){
		double dist = norm(projections.col(i), q, NORM_L2);
		if (dist < mindist){
			mindist = dist;
		}
	}
	//x = clock() - x;
	//printf("%lf\n", x / CLOCKS_PER_SEC);

	//Mat s;
	//if (sample.type() != CV_32F)
	//	sample.convertTo(s, CV_32F);

	//double x = clock();
	//float smpl[4096];
	//float qq[19];

	//for (int i = 0; i < 4096; ++i) {
	//	smpl[i] = ((float *)s.data)[i] - ((float *)mean.data)[i];
	//}

	//Mat e = eigens.t();
	//float *eig = (float *)e.data;

	//for (int i = 0; i < e.rows; ++i) {
	//	qq[i] = 0;
	//	for (int j = 0; j < e.cols; ++j) {
	//		qq[i] += smpl[j] * eig[i * e.cols + j];
	//	}
	//}

	//Mat pprojs = projections.t();
	//float *projs = (float *)pprojs.data;

	//double mindist = DBL_MAX;
	//for (int i = 0; i < pprojs.rows; ++i) {
	//	float dist = 0;
	//	for (int j = 0; j < pprojs.cols; ++j) {
	//		dist += (qq[j] - projs[i * pprojs.cols + j]) * (qq[j] - projs[i * pprojs.cols + j]);
	//	} 

	//	dist  = sqrtf(dist);
	//	if (dist < mindist) mindist = dist;
	//}

	return mindist;
}

