// system includes
#include <iostream>
#include <vector>
#include <ctime>
#include <Windows.h>

// opencv includes
#include <opencv2\core\core.hpp>

// project includes
#include "utilities.h"
#include "pca.h"

using namespace std;


// eksperiment 1
int main(){ 
	vector<Mat> data;
	vector<int> labels;

	double score = 0;
	int vectors = 20;

	double time;

	cout << "Ucitavam podatke..." << endl;
	//load("../lica.txt", data, labels, FACE);
	load("../dlanovi.txt", data, labels, PALM);

	cout << "Unakrsna validacija (4-fold)..." << endl;
	for (int i = 0; i < 4; ++i){
		time = clock();

		cout << "Iteracija " << (i + 1) << endl;
		Mat train, test;
		vector<int> trainLabels, testLabels;

		cout << "\tDijelim podake na 2 skupa (5 ucenje/15 testiranje)..." << endl;
		splitData(i, data, labels, train, trainLabels, test, testLabels);
	
		cout << "\tProvodim PCA koristeci " << (vectors == 0 ? train.cols : vectors) << " svojstvenih vektora..." << endl;
		pca p(train, trainLabels, vectors);

		cout << "\tTestiram..." << endl;
		double perc = p.test(test, testLabels);
		stringstream s;
		s << perc;
		cout << "\tTocnost: ";
		colorText(s.str());
		cout << endl << endl;

		score += perc;
		time += (clock() - time);
	}

	stringstream s;
	s << score/4;
	cout << endl << "Prosjecna tocnost: ";
	colorText(s.str());
	cout << endl << endl;

	cout << "Prosjecno vrijeme po jednoj iteraciji unakrsne validacije: " << time / 4 / CLOCKS_PER_SEC << endl;

	system("pause");
	return 0;
}