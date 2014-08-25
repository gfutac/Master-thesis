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
	vector<int> labels;

	Mat eigens;
	Mat projections;

	void calculate();
	void calcMean();

	Mat project(const Mat &sample);
	double minDistance(const Mat &data);
public:
	pca(const Mat &data, vector<int> &labels, int vectors);
	vector<double> distances(Mat &testData, vector<int> &testLabels);
};

#endif PCA_H
