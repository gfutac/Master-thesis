#ifndef PCA_H
#define PCA_H

#include <opencv2\core\core.hpp>
#include <opencv2\gpu\gpu.hpp>

using namespace cv;

class pca{
private: 
	Mat data;
	int vectors;
	int label;


	void calculate();
	void calcMean();

	Mat project(const Mat &sample);
	
public:
	pca(const Mat &data, int label, int vectors);
	double minDistance(const Mat &data);
	int getLabel();

	Mat mean;
	Mat eigens;
	Mat projections;

};

#endif PCA_H
