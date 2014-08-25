#include <iostream>
#include <ctime>
#include <Windows.h>

#include "utilities.h"
#include "pca.h"
#include "loadBar.h"

using namespace std;

// eksperiment 3
int main(){
	srand((unsigned)time(0));

	map<int, vector<Mat>> data;
	int vectors = 20;

	double score = 0;
	double time;

	cout << "Ucitavam podatke..." << endl;
	load("../dlanovi.txt", data, PALM);

	cout << "Unakrsna validacija (leave-one-out 19/1)..." << endl;
	for (int iter = 0; iter < 20; ++iter) {
		double time = clock();

		cout << "Iteracija " << (iter + 1) << endl;
		map<int, Mat> train, test;

		cout << "\tDijelim podake na 2 skupa (19 ucenje/1 testiranje)..." << endl;
		splitData(iter, data, train, test);

		vector<pca> pcas;
		cout << "\tProvodim PCA koristeci " << vectors << " svojstvenih vektora za svaki razred..." << endl;
		int t = 0;
		for (auto c = train.begin(); c != train.end(); ++c){
			loadBar(30, (float)t++/(train.size() - 1));

			int label = c->first;
			Mat tdata = c->second;
	
			pca p(tdata, label, vectors);
			pcas.push_back(p);
		}

		cout << "\tTestiram..." << endl;

		t = 0;
		double perc = 0;
		int trues = 0, total = 0;
		for (auto c = test.begin(); c != test.end(); ++c){
			loadBar(30, (float)t++/(test.size() - 1));

			int label = c->first;
			Mat tdata = c->second;

			vector<double> distances;
			for (int i = 0; i < pcas.size(); ++i){
				pca p = pcas[i];
				double d = p.minDistance(tdata);
				distances.push_back(d);
			}

			int idx = min_element(distances.begin(), distances.end()) - distances.begin();
			int predicted = pcas[idx].getLabel();
			if (predicted == label) ++trues;
			++total;
		}

		perc = (double)trues/total;	
		stringstream s;
		s << perc;

		cout << "\tTocnost: ";
		colorText(s.str());
		cout << endl << endl;

		score += perc;
		time += (clock() - time);
	}

	stringstream s;
	s << score/20;
	cout << endl << "Prosjecna tocnost: ";
	colorText(s.str());
	cout << endl << endl;

	cout << "Prosjecno vrijeme po jednoj iteraciji unakrsne validacije: " << time / 20 / CLOCKS_PER_SEC << endl;
	
	system("pause");
	return 0;
}

