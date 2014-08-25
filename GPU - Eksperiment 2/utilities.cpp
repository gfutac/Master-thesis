#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <ctime>
#include <Windows.h>

#include <opencv2\core\core.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "utilities.h"

using namespace std;
using namespace cv;
using namespace gpu;


/*
	Helper function for string splitter
*/
vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while(getline(ss, item, delim)) {
        elems.push_back(item);
    }

    return elems;
}

/*
	Splits the string by the given delimiter
*/
vector<string> split(const string &s, char delim) {
    vector<string> elems;
    return split(s, delim, elems);
}

void colorText(string s){
	HANDLE console;
	CONSOLE_SCREEN_BUFFER_INFO info;
	console = GetStdHandle(STD_OUTPUT_HANDLE);
	GetConsoleScreenBufferInfo(console, &info);

	// print colored text
	SetConsoleTextAttribute(console, 10);
	cout << s;

	// reset color
	SetConsoleTextAttribute(console, info.wAttributes);
}


//enum imgType { FACE = 1, PALM = 2 };

/* 
	Preprocesses the image; 
	If image is a face image then select face ROI, resize it to 64x64 and make and return a column vector.
	If image is a palm image then just resize it to 64x64 and make and return a column vector.
*/
Mat impreprocess(Mat &img, int type){
	Mat vector;

	int size = 64;
	
	//if (img.type() != CV_32F)
	//	img.convertTo(img, CV_32F);

	if (type == FACE){
		Rect roi(40,50,100,130);
		vector = img(roi);
		resize(vector, vector, Size(size, size));	
	} else if (type == PALM){
		resize(img, vector, Size(size, size));
		//equalizeHist (vector, vector);	
	}

	vector = vector.reshape(0, size * size);
	return vector;
}

/*
	Form a matrix from a given vector<> of column vectors (Mat)
*/
Mat asMatrix(const vector<Mat> &src){
	int rows = src[0].rows;
	int cols = src.size();
	int type = DataType<int>::type;
	//int type = src[0].type();

	Mat m(rows, cols, type);
	for (int i = 0; i < cols; ++i){
		src[i].copyTo(m.col(i));
	}

	return m;
}

/*
	Loads all data in one matrix
*/
void load(string path, map<int, vector<Mat>> &data, int type){
	fstream infile(path);
	if (!infile.is_open()){
		cerr << "Error reading input file" << endl;
		system("pause");
		exit(1);
	}

	string line;
	while (getline(infile, line)){
		istringstream iss(line);
		vector<string> t = split(line, ';');

		string file = t[0];
		int label = atoi(t[1].c_str());

		Mat img = imread(file, 0);
		Mat sample = impreprocess(img, type);	

		data[label].push_back(sample);
	}
}

void splitData(int iter, map<int, vector<Mat>> &data, map<int, Mat> &train, map<int, Mat> &test) {
	for (auto i = data.begin(); i != data.end(); ++i){
		vector<Mat> tmp;
		for (int j = 0; j < i->second.size(); ++j){
			if (j == iter){
				test[i->first] = i->second[j];
			}
			else{
				tmp.push_back(i->second[j]);
			}
		}

		train[i->first] = asMatrix(tmp);
	}	
}