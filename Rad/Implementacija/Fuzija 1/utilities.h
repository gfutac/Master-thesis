#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <ctime>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

enum imgType { FACE = 1, PALM = 2 };
vector<string> &split(const string &s, char delim, vector<string> &elems);
vector<string> split(const string &s, char delim);
Mat impreprocess(Mat &img, int type);
Mat asMatrix(const vector<Mat> &src);
void load(string path, vector<Mat> &data, vector<int> &labels, int type);
void splitData(int iter, vector<Mat> data, vector<int> labels, Mat &trainData, vector<int> &trainLabels, Mat &testData, vector<int> &testLabels);
void colorText(string s);
void distanceNormalize(double *arr, int size);

#endif UTILITIES_H