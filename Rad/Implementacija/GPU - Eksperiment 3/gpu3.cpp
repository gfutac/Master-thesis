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

extern "C" void foo(DevMem2Df, float[]);
extern "C" void getMinDistances(DevMem2Df, DevMem2Df, DevMem2Df, DevMem2Di, float *, int, float *);

#define N 4096

int main(){
	colorText("GPU - Eksperiment 3\n\n");
	cudaDeviceReset();

	map<int, vector<Mat>> data;
	int vectors = 169;
	int templateSize = 19;
	double score = 0;
	double time;

	float *eigens, *projections, *means;

	cout << "Ucitavam podatke..." << endl;
	load("D:\\Futac\\Diplomski rad\\Implementacija\\diplomski-rad-futac-goran\\dlanovi.txt", data, PALM); // v50/t10 = 0.9888411

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

		int rows = train.size();

		if (iter == 0) {
			eigens = new float[rows * (N * vectors)];
			projections = new float[rows * (vectors * templateSize)];
			means = new float[rows * N];
		}

		int bar = 0;
		for (auto c = train.begin(); c != train.end(); ++c){
			loadBar(30, (float)bar++/(train.size() - 1));
			int label = c->first;
			Mat tdata = c->second;
	
			pca p(tdata, label, vectors, templateSize);
			pcas.push_back(p);

			Mat a = p.eigens.t();
			std::copy((float *)a.data, &(*((float *)a.data + N * vectors)), &eigens[N * vectors * t]);

			a = p.projections.t();
			std::copy((float *)a.data, &(*((float *)a.data + vectors * templateSize)), &projections[vectors * templateSize * t]);

			a = p.mean;
			std::copy((float *)a.data, &(*((float *)a.data + N)), &means[N * t++]);
		}

		Mat eigensMat(rows, N * vectors, CV_32F, eigens);
		Mat projectionsMat(rows, vectors * templateSize, CV_32F, projections);
		Mat meansMat(rows, N, CV_32F, means);

		GpuMat e(eigensMat);
		GpuMat p(projectionsMat);
		GpuMat m(meansMat);

		vector<Mat> a;
		vector<int> labels;
		for (auto c = test.begin(); c != test.end(); ++c) {
			a.push_back(c->second);
			labels.push_back(c->first);
		}

		Mat samplesMat = asMatrix(a).t();
		GpuMat s(samplesMat);

		vector<float> distances(train.size() * train.size());
		vector<float> minIndices(train.size());

		cout << "\tTestiram..." << endl;
		getMinDistances(e, p, m, s, &distances[0], train.size(), &minIndices[0]);

		double perc = 0;
		int trues = 0, total = 0;
		for (int i = 0; i < train.size(); ++i){
			vector<float> d(distances.begin() + i * train.size(), distances.begin() + (i + 1) * train.size());
			int idx = std::min_element(d.begin(), d.end()) - d.begin();
			int predicted = pcas[idx].getLabel();
			if (predicted == labels[i]) ++trues;
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

	delete [] eigens;
	delete [] projections;
	delete [] means;

	stringstream s;
	s << score/20;
	cout << endl << "Prosjecna tocnost: ";
	colorText(s.str());
	cout << endl << endl;
	
	std::system("pause");
	cudaDeviceReset();
	return 0;
}

