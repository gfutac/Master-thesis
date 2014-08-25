#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <ctime>
#include <Windows.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;


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


enum imgType { FACE = 1, PALM = 2 };

/* 
	Preprocesses the image; 
	If image is a face image then select face ROI, resize it to 64x64 and make and return a column vector.
	If image is a palm image then just resize it to 64x64 and make and return a column vector.
*/
Mat impreprocess(Mat &img, int type){
	Mat vector;

	int size = 64;

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
	int type = src[0].type();

	Mat m(rows, cols, type);
	for (int i = 0; i < cols; ++i){
		src[i].copyTo(m.col(i));
	}

	return m;
}

/*
	Loads all data in one matrix
*/
void load(string path, vector<Mat> &data, vector<int> &labels, int type){
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

		data.push_back(sample);
		labels.push_back(label);
	}
}

void indices(vector<int> &tr, vector<int> &te, int iter){
	for (int j = 0; j < 20; ++j) {
		if (((j - iter) % 4) == 0){
			tr.push_back(j);
		} else { 
			te.push_back(j);
		}
	}
}

void splitData(int iter, vector<Mat> data, vector<int> labels, Mat &trainData, vector<int> &trainLabels, Mat &testData, vector<int> &testLabels) {
	vector<int> tr, te;
	indices(tr, te, iter);	

	vector<Mat> train, test;

	for (int i = 0; i < data.size(); i += 20){
		for (int j = 0; j < tr.size(); ++j) { train.push_back(data[i + j]); trainLabels.push_back(labels[i + j]); }
		for (int j = 0; j < te.size(); ++j) { test.push_back(data[i + j]); testLabels.push_back(labels[i + j]); }
	}

	trainData = asMatrix(train);
	testData = asMatrix(test);
}