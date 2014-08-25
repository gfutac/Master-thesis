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

	//printf("block = [%d]; thread[%d]\n", idx, tid);
	
	int i, j, k;
	i = j = k = 0;

	__shared__ float p[VECTORS * VECTORS];

	if (tid == 0){
		for (i = 0; i < VECTORS * VECTORS; ++i) {
			p[i] = projections(idx)[i];
		}
	}

	__syncthreads();

	float sample[N];
	float *mean = means(idx);

	int4 *ss = (int4 *)samples(tid);
	float4 *m = (float4 *)means(idx);
	
	for (i = 0; i < N/4; i += 1){
		sample[i*4] = ss[i].x - m[i].x;
		sample[i*4+1] = ss[i].y - m[i].y;
		sample[i*4+2] = ss[i].z - m[i].z;
		sample[i*4+3] = ss[i].w - m[i].w;
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
			res += (proj[j] - p[i * VECTORS + j]) * (proj[j] - p[i * VECTORS + j]);
		}

		res = sqrtf(res);
		if (res < min) min = res;
	}

	distances[tid * SIZE + idx] = min;

}

extern "C" float getMinDistances(DevMem2Df eigens, DevMem2Df projections, DevMem2Df means, DevMem2Di samples, float *distances, int size){
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

	calculateDistances<<<size, size>>>(eigens, projections, means, samples, d_distances);

	cudaEventRecord(event2, 0);

	//synchronize
	cudaEventSynchronize(event1); //optional
	cudaEventSynchronize(event2); //wait for the event to be executed!

	//calculate time
	float dt_ms;
	cudaEventElapsedTime(&dt_ms, event1, event2);

	cudaMemcpy(distances, d_distances, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_distances);
	cudaFree(dsample);

	return dt_ms;
}
