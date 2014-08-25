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
	// lica
	double pr1 = 0.984 / (0.984 + 0.917);
	// dlanovi
	double pr2 = 1 - pr1;

	vector<Mat> faceData, palmData;
	vector<int> labels;

	double score = 0;
	int vectors = 50;

	double time;

	cout << "Ucitavam podatke..." << endl;
	load("../lica.txt", faceData, labels, FACE);
	load("../dlanovi.txt", palmData, vector<int>(), PALM);

	cout << "Unakrsna validacija (4-fold)..." << endl;
	for (int i = 0; i < 4; ++i){
		time = clock();

		cout << "Iteracija " << (i + 1) << endl;
		Mat faceTrain, faceTest;
		Mat palmTrain, palmTest;

		vector<int> trainLabels, testLabels;

		cout << "\tDijelim podake na 2 skupa (5 ucenje/15 testiranje)..." << endl;
		splitData(i, faceData, labels, faceTrain, trainLabels, faceTest, testLabels);
		splitData(i, palmData, labels, palmTrain, vector<int>(), palmTest, vector<int>());

		cout << "\tProvodim PCA koristeci " << (vectors == 0 ? faceTrain.cols : vectors) << " svojstvenih vektora..." << endl;
		pca faces(faceTrain, trainLabels, vectors);
		pca palms(palmTrain, trainLabels, vectors);

		cout << "\tTestiram..." << endl;
		vector<double> faceDistances, palmDistances;
		faceDistances = faces.distances(faceTest, testLabels);
		palmDistances = palms.distances(palmTest, testLabels);

		distanceNormalize(&faceDistances[0], faceDistances.size());
		distanceNormalize(&palmDistances[0], palmDistances.size());

		vector<double> tsm;
		for (int i = 0; i < faceDistances.size(); ++i){
			tsm.push_back(pr1 * faceDistances[i] + pr2 * palmDistances[i]);
		}

		int idx = max_element(tsm.begin(), tsm.end()) - tsm.begin();
		int label = trainLabels[idx];



		//stringstream s;
		//s << perc;
		//cout << "\tTocnost: ";
		//colorText(s.str());
		//cout << endl << endl;

		//score += perc;
		//time += (clock() - time);
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