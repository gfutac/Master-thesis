#ifndef PCA_H
#define PCA_H

#include <opencv2\core\core.hpp>
#include <opencv2\gpu\gpu.hpp>

using namespace cv;

class pca{
private: 
	Mat data;
	Mat mean;
	int vectors;
	int label;

	Mat eigens;
	Mat projections;

	void calculate();
	void calcMean();

	Mat project(const Mat &sample);
	
public:
	pca(const Mat &data, int label, int vectors);
	double minDistance(const Mat &data);
	int getLabel();
};

#endif PCA_H
