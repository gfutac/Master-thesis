#include <iostream>

#include "pca.h"
#include "loadBar.h"
#include <opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;

pca::pca(const Mat &data, vector<int> &labels, int vectors = 0) : data(data), vectors(vectors), labels(labels) {
	if (vectors == 0){
		vectors = data.cols - 1;
	}
	calculate();
}

double pca::test(Mat &testData, vector<int> &testLabels){
	int trues = 0;
	int total = 0;

	for (int i = 0; i < testData.cols; ++i){
		loadBar(20, (float)i/(testData.cols - 1));

		Mat sample = testData.col(i);
		int idx = minDistanceIndex(sample);
		int label = labels[idx];

		if (label == testLabels[i]) ++trues;
		++total;
	}

	return (double)trues/total;
}

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
int pca::minDistanceIndex(const Mat &sample){
	Mat q = project(sample);

	double mindist = DBL_MAX;
	int minIdx = 0;
	for (int i = 0; i < projections.cols; ++i){
		double dist = norm(projections.col(i), q, NORM_L2);
		if (dist < mindist){
			mindist = dist;
			minIdx = i;
		}
	}

	return minIdx;
}

