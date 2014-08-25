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
	int templateSize;

	void calculate();
	void calcMean();
	
public:
	Mat mean;
	Mat eigens;
	Mat projections;

	Mat project(const Mat &sample);
	pca(const Mat &data, int label, int vectors, int templateSize);
	double minDistance(const Mat &data);
	int getLabel();
};

#endif PCA_H
