#include <iostream>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2\core\core.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <opencv2\core\cuda_devptrs.hpp>

#include <thrust\host_vector.h>

#include "utilities.h"
#include "pca.h"
#include "loadBar.h"

using namespace std;
using namespace cv;
using namespace gpu;
using namespace thrust;

extern "C" float getMinDistances(DevMem2Df, DevMem2Df, DevMem2Df, DevMem2Di, float *, int);

#define N 4096

int main(){
	cudaDeviceReset();
	colorText("GPU - Fuzija 2\n\n");

	// lica
	double pr1 = 0.8225 / (0.8225 + 0.7337);
	// dlanovi
	double pr2 = 1 - pr1;
	
	map<int, vector<Mat>> faceData, palmData;

	int vectors = 19;

	double score = 0;
	double time;

	float *feigens, *fprojections, *fmeans;
	float *peigens, *pprojections, *pmeans;

	cout << "Ucitavam podatke..." << endl;
	load("D:\\Futac\\Diplomski rad\\Implementacija\\diplomski-rad-futac-goran\\lica.txt", faceData, FACE);
	load("D:\\Futac\\Diplomski rad\\Implementacija\\diplomski-rad-futac-goran\\dlanovi.txt", palmData, PALM);

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

		int rows = trainFaces.size();

		if (iter == 0){ 
			feigens = new float[rows * (N * vectors)];
			fprojections = new float[rows * (vectors * vectors)];
			fmeans = new float[rows * N];

			peigens = new float[rows * (N * vectors)];
			pprojections = new float[rows * (vectors * vectors)];
			pmeans = new float[rows * N];
		}

		
		vector<int> trainLabels;
		for (auto c = trainFaces.begin(), d = trainPalms.begin(); c != trainFaces.end(), d != trainPalms.end(); ++c, ++d){
			// lice
			int label = c->first;
			Mat tdata = c->second;
			pca p(tdata, label, vectors);
			trainLabels.push_back(label);

			Mat a = p.eigens.t();
			//std::copy((float *)a.data, &(*((float *)a.data + N * vectors)), &feigens[N * vectors * t]);
			memcpy(&feigens[N * vectors * t], a.data, N * vectors * sizeof(float));
			a = p.projections.t();
			//std::copy((float *)a.data, &(*((float *)a.data + vectors * vectors)), &fprojections[vectors * vectors * t]);
			memcpy(&fprojections[vectors * vectors * t], a.data, vectors * vectors * sizeof(float));
			a = p.mean;
			//std::copy((float *)a.data, &(*((float *)a.data + N)), &fmeans[N * t]);
			memcpy(&fmeans[N * t], a.data, N * sizeof(float));

			// dlanovi
			tdata = d->second;
			pca pp(tdata, label, vectors);

			a = pp.eigens.t();
			std::copy((float *)a.data, &(*((float *)a.data + N * vectors)), &peigens[N * vectors * t]);
			a = pp.projections.t();
			std::copy((float *)a.data, &(*((float *)a.data + vectors * vectors)), &pprojections[vectors * vectors * t]);
			a = pp.mean;
			std::copy((float *)a.data, &(*((float *)a.data + N)), &pmeans[N * t++]);		
		}

		Mat feigensMat(rows, N * vectors, CV_32F, feigens);
		Mat fprojectionsMat(rows, vectors * vectors, CV_32F, fprojections);
		Mat fmeansMat(rows, N, CV_32F, fmeans);

		GpuMat fe(feigensMat);
		GpuMat fp(fprojectionsMat);
		GpuMat fm(fmeansMat);

		Mat peigensMat(rows, N * vectors, CV_32F, peigens);
		Mat pprojectionsMat(rows, vectors * vectors, CV_32F, pprojections);
		Mat pmeansMat(rows, N, CV_32F, pmeans);

		GpuMat pe(peigensMat);
		GpuMat pp(pprojectionsMat);
		GpuMat pm(pmeansMat);


		vector<Mat> fa, pa;
		vector<int> testLabels;
		for (auto c = testFaces.begin(), d = testPalms.begin(); c != testFaces.end(), d != testPalms.end(); ++c, ++d){
			fa.push_back(c->second);
			pa.push_back(d->second);

			testLabels.push_back(c->first);
		}

		Mat fsamplesMat = asMatrix(fa).t();
		GpuMat fs(fsamplesMat);

		Mat psamplesMat = asMatrix(pa).t();
		GpuMat ps(psamplesMat);

		vector<float> fdistances(trainFaces.size() * trainFaces.size());
		vector<float> pdistances(trainFaces.size() * trainFaces.size());

		cout << "\tTestiram..." << endl;
		float elapsedTime = getMinDistances(fe, fp, fm, fs, &fdistances[0], trainFaces.size());
		elapsedTime += getMinDistances(pe, pp, pm, ps, &pdistances[0], trainPalms.size());
		cout << "\tUtroseno vrijeme: " << elapsedTime/1000 << endl;

		double perc = 0;
		int trues = 0, total = 0;
		for (int i = 0; i < trainFaces.size(); ++i){
			vector<float> fd(fdistances.begin() + i * trainFaces.size(), fdistances.begin() + (i + 1) * trainFaces.size());
			vector<float> pd(pdistances.begin() + i * trainPalms.size(), pdistances.begin() + (i + 1) * trainPalms.size());

			distanceNormalize(&fd[0], fd.size());
			distanceNormalize(&pd[0], pd.size());

			vector<double> tsm;
			for (int i = 0; i < fd.size(); ++i){
				tsm.push_back(pr1 * fd[i] + pr2 * pd[i]);
			}

			int idx = std::max_element(tsm.begin(), tsm.end()) - tsm.begin();
			int predicted = trainLabels[idx];

			if (predicted == testLabels[i]) ++trues;
			++total;
		}

		perc = (double)trues/total;	
		stringstream ss;
		ss << perc;

		cout << "\tTocnost: ";
		colorText(ss.str());
		cout << endl << endl;

		score += perc;
	}

	delete [] feigens;
	delete [] fprojections;
	delete [] fmeans;
	delete [] peigens;
	delete [] pprojections;
	delete [] pmeans;

	stringstream s;
	s << score/20;
	cout << endl << "Prosjecna tocnost: ";
	colorText(s.str());
	cout << endl << endl;
	
	std::system("pause");
	cudaDeviceReset();
	return 0;
}

