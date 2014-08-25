#include <opencv2\gpu\devmem2d.hpp>

#include <cuda_runtime.h>

#include <thrust\device_vector.h>

using namespace cv::gpu;
using namespace thrust;

#define N 4096
#define VECTORS 19
#define PROJECTIONS 19
#define SIZE 151

__global__ void calculateDistances(DevMem2Df eigens, DevMem2Df projections, DevMem2Df means, DevMem2Di samples, float *distances){
	int idx = blockIdx.x;
	int tid = threadIdx.x;
	
	int i, j, k;
	i = j = k = 0;

	__shared__ float p[VECTORS * VECTORS];

	if (tid == 0){
		for (i = 0; i < VECTORS * VECTORS; ++i) {
			p[i] = projections(idx)[i];
		}
	}

	float sample[N];
	
	float *mean = means(idx);
	for (i = 0; i < N; i+=4){
		sample[i] = samples(tid)[i] - mean[i];
		sample[i+1] = samples(tid)[i+1] - mean[i+1];
		sample[i+2] = samples(tid)[i+2] - mean[i+2];
		sample[i+3] = samples(tid)[i+3] - mean[i+3];
	}

	// projicirati uzorak
	float proj[VECTORS];
	float *eig = eigens(idx);

	for (i = 0; i < VECTORS; ++i) {
		proj[i] = 0;
		for (j = 0; j < N; j += 4) {
			proj[i] += sample[j] * eig[i * N + j] +
				sample[j+1] * eig[i * N + j+1] + 
				sample[j+2] * eig[i * N + j+2] +
				sample[j+3] * eig[i * N + j+3];
		}
	}

	// usporediti sa svakim projiciranim i dobiti najmanju udaljenost
	float min = FLT_MAX;
	for (i = 0; i < PROJECTIONS; ++i) {
		float res = 0;
		for (j = 0; j < VECTORS; ++j) {
			//res = res + fabsf(proj[j] - p[i * VECTORS + j]);
			res += (proj[j] - p[i * PROJECTIONS + j]) * (proj[j] - p[i * PROJECTIONS + j]);
		}

		res = sqrtf(res);
		if (res < min) min = res;
	}

	distances[tid * SIZE + idx] = min;

}

extern "C" void getMinDistances(DevMem2Df eigens, DevMem2Df projections, DevMem2Df means, DevMem2Di samples, float *distances, int size, int *minIndices){
	float *dsample, *d_distances; 
	int *d_minindices;

	cudaMalloc((float **)&d_distances, sizeof(float) * size * size);
	cudaMalloc((float **)&dsample, sizeof(float) * N);
	cudaMalloc((int **)&d_minindices, sizeof(int) * size);

	cudaEvent_t event1, event2;
cudaEventCreate(&event1);
cudaEventCreate(&event2);

//record events around kernel launch
cudaEventRecord(event1, 0); //where 0 is the default stream
//cudaFuncSetCacheConfig("calculateDistances", cudaFuncCachePreferShared);
	calculateDistances<<<size, size>>>(eigens, projections, means, samples, d_distances);
	cudaEventRecord(event2, 0);

//synchronize
cudaEventSynchronize(event1); //optional
cudaEventSynchronize(event2); //wait for the event to be executed!

//calculate time
float dt_ms;
cudaEventElapsedTime(&dt_ms, event1, event2);

	std::cout << "\tUtroseno vrijeme: " << dt_ms/1000 << std::endl;

	//cudaMemcpy(minIndices, d_minindices, sizeof(int) * size, cudaMemcpyDeviceToHost);

	cudaMemcpy(distances, d_distances, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_distances);
	cudaFree(dsample);
}
