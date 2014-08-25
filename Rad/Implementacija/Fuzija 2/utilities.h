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
void colorText(string s);

Mat impreprocess(Mat &img, int type);
Mat asMatrix(const vector<Mat> &src);
void load(string path, map<int, vector<Mat>> &data, int type);
void splitData(int iter, map<int, vector<Mat>> &data, map<int, Mat> &train, map<int, Mat> &test);
void colorText(string s);
void distanceNormalize(double *arr, int size);

#endif UTILITIES_H