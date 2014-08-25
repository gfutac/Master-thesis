#include <iostream>
#include <ctime>
#include <Windows.h>

#include "utilities.h"
#include "pca.h"
#include "loadBar.h"

using namespace std;

// eksperiment 3
int main(){
	colorText("Eksperiment 3\n\n");
	//srand((unsigned)time(0));

	map<int, vector<Mat>> data;

	double score = 0;
	double time = 0;

	cout << "Ucitavam podatke..." << endl;
	load("../lica.txt", data, FACE);

	int vectors = 20;
	int templateSize = 19;

	cout << "Unakrsna validacija (leave-one-out 19/1)..." << endl;
	for (int iter = 0; iter < 1; ++iter) {

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
	
			pca p(tdata, label, vectors, templateSize);
			pcas.push_back(p);
		}

		cout << "\tTestiram..." << endl;

		t = 0;
		double perc = 0;
		int trues = 0, total = 0;
		
		//double iterTime = clock();
		for (auto c = test.begin(); c != test.end(); ++c){
			loadBar(30, (float)t++/(test.size() - 1));

			int label = c->first;
			Mat tdata = c->second;

			double iterTime = clock();
			vector<double> distances;
			for (int i = 0; i < 10; ++i){
				distances.clear();
				for (int i = 0; i < pcas.size(); ++i){
					pca p = pcas[i];
					double d = p.minDistance(tdata);
					distances.push_back(d);
				}

				double t = (clock() - iterTime) / CLOCKS_PER_SEC;
				printf("\tUtroseno vrijeme %d: %lfs\n", (i+1), t);
			}

			int idx = min_element(distances.begin(), distances.end()) - distances.begin();
			int predicted = pcas[idx].getLabel();
			if (predicted == label) ++trues;
			++total;

			break;
		}
		//iterTime = (clock() - iterTime) / CLOCKS_PER_SEC;
		
		perc = (double)trues/total;	
		stringstream s;
		s << perc;

		cout << "\tTocnost: ";
		colorText(s.str());
		cout << endl;
		//printf("\tUtroseno vrijeme: %lfs\n", iterTime);

		score += perc;
		//time += iterTime;
	}

	stringstream s;
	s << score/20;
	cout << endl << "Prosjecna tocnost: ";
	colorText(s.str());
	cout << endl << endl;

	cout << "Prosjecno vrijeme po jednoj iteraciji unakrsne validacije: " << time / 20 << endl;
	
	system("pause");
	return 0;
}

