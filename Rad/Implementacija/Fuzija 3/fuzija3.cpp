#include <iostream>
#include <ctime>
#include <Windows.h>

#include "utilities.h"
#include "pca.h"
#include "loadBar.h"

using namespace std;

// fuzija 3
int main(){
	colorText("Fuzija 3\n\n");
	// lica
	double pr1 = 0.999 / (0.999 + 0.999);
	// dlanovi
	double pr2 = 1 - pr1;

	map<int, vector<Mat>> faceData, palmData;
	double score = 0;
	double time;
	int vectors = 20;
	
	cout << "Ucitavam podatke..." << endl;
	load("../lica.txt", faceData, FACE);
	load("../dlanovi.txt", palmData, PALM);

	cout << "Unakrsna validacija (leave-one-out 19/1)..." << endl;
	for (int iter = 0; iter < 20; ++iter) {
		double time = clock();

		cout << "Iteracija " << (iter + 1) << endl;
		map<int, Mat> trainFaces, testFaces;
		map<int, Mat> trainPalms, testPalms;

		cout << "\tDijelim lica na 2 skupa (19 ucenje/1 testiranje)..." << endl;
		splitData(iter, faceData, trainFaces, testFaces);

		cout << "\tDijelim dlanovi na 2 skupa (19 ucenje/1 testiranje)..." << endl;
		splitData(iter, palmData, trainPalms, testPalms);
		
		vector<pca> faces, palms;
		cout << "\tProvodim PCA koristeci " << vectors << " svojstvenih vektora za svaki razred..." << endl;
		int t = 0;

		for (auto c = trainFaces.begin(), d = trainPalms.begin(); c != trainFaces.end(), d != trainPalms.end(); ++c, ++d){
			loadBar(30, (float)t++/(trainFaces.size() - 1));

			int label = c->first;
			Mat tdata = c->second;
			pca p(tdata, label, vectors);
			faces.push_back(p);

			label = d->first;
			tdata = d->second;
			pca pp(tdata, label, vectors);
			palms.push_back(pp);
		}

		cout << "\tTestiram..." << endl;

		t = 0;
		double perc = 0;
		int trues = 0, total = 0;
		double iterTime = clock();
		for (auto c = testFaces.begin(), d = testPalms.begin(); c != testFaces.end(), d != testPalms.end(); ++c, ++d){
			loadBar(30, (float)t++/(testFaces.size() - 1));

			int label = c->first;
			Mat fdata = c->second;
			Mat pdata = d->second;

			vector<double> faceDistances, palmDistances;

			for (int i = 0; i < faces.size(); ++i){
				pca p = faces[i];
				double d = p.minDistance(fdata);
				faceDistances.push_back(d);

				p = palms[i];
				d = p.minDistance(pdata);
				palmDistances.push_back(d);
			}

			distanceNormalize(&faceDistances[0], faceDistances.size());
			distanceNormalize(&palmDistances[0], palmDistances.size());

			vector<double> tsm;
			for (int i = 0; i < faceDistances.size(); ++i){
				tsm.push_back(pr1 * faceDistances[i] + pr2 * palmDistances[i]);
			}

			int idx = max_element(tsm.begin(), tsm.end()) - tsm.begin();
			int predicted = faces[idx].getLabel();

			if (predicted == label) ++trues;
			++total;
		}
		iterTime = clock() - iterTime;
		cout << "\tVrijeme: " << iterTime / CLOCKS_PER_SEC << endl;

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

