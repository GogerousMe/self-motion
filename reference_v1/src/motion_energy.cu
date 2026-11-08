#include "motion_energy.h"

#include <stdio.h>	// printf
#include <cassert> // assert
#include <cmath>   // expf
#include <cfloat>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <algorithm>

#include <cuda_definitions.h>


#define IMUL(a, b) __mul24(a, b)
#define iDivUp(a,b) ((a)+(b)-1)/(b)
#define CONV1_THREAD_SIZE 256
#define CONVN_THREAD_SIZE1 8
#define CONVN_THREAD_SIZE2 128 // supports OMS kernels up to 121 while staying within 1024 threads/block

// 28 space-time orientations of V1 simple cells
#define nrFilters 28

// 8 directions
#define nrDirs 8

// each of these is a unit vector (e.g. d_v1popDirs[0][1..3] => (-0.6559)^2+(0.7246)^2+(0.2113)^2 == 1
// all vectors lie on the surface of a dome (hemisphere) in 3D Fourier space
// x, y, t
__constant__ float d_v1popDirs[3][nrFilters] = {
	{ 0.7246,-0.9718, 0.7496,-0.5837,-0.0810, 0.9439, 0.3203,-0.8712,-0.1593,-0.5142, 0.9304, 0.3737,-0.8031,-0.8126, 0.6004,-0.5738, 0.0024, 0.5969, 0.1436, 0.7757,-0.4004,-0.5108, 0.2375,-0.2221,-0.5140, 0.5194,-0.0870, 0.3838},
	{-0.6559,-0.1019, 0.6240,-0.7797, 0.9692,-0.2312,-0.9151, 0.4207,-0.9533, 0.8175, 0.2398, 0.8810,-0.4430, 0.0588,-0.5384, 0.5644, 0.7931, 0.5142,-0.7680,-0.0669,-0.6670,-0.2747, 0.5034, 0.5042, 0.1580, 0.1332,-0.5159,-0.3549},
	{ 0.2113, 0.2126, 0.2210, 0.2266, 0.2327, 0.2359, 0.2451, 0.2529, 0.2567, 0.2593, 0.2772, 0.2902, 0.3984, 0.5799, 0.5913, 0.5935, 0.6091, 0.6160, 0.6241, 0.6275, 0.6283, 0.8146, 0.8308, 0.8345, 0.8431, 0.8441, 0.8522, 0.8525}
};

// \TODO use dynamic shared memory to allocate interpolation weights from filters onto directions
//extern __shared__ float motionProjDyn[];

__constant__ float motionProj[3][nrFilters][nrDirs] = {
// 1.5 px/fr
{	{ 0.002719,  0.011644, -0.002266, -0.094267, -0.088188, -0.021185, -0.097296, -0.081224},
	{-0.023337, -0.106719, -0.077625,  0.007519,  0.015789, -0.006119, -0.100403, -0.080257},
	{ 0.002680, -0.081351, -0.101069, -0.017226, -0.080847, -0.101749, -0.007590,  0.013499},
	{-0.105574, -0.075236,  0.004742,  0.012976, -0.014051, -0.107587, -0.074961, -0.019999},
	{-0.101953, -0.078081, -0.011287, -0.098204, -0.084890,  0.000210,  0.010038, -0.012016},
	{ 0.013383,  0.006850, -0.065943, -0.111274, -0.019242, -0.057148, -0.114513, -0.024744},
	{-0.061140,  0.005103,  0.009873, -0.040867, -0.119077, -0.040599, -0.024648, -0.118630},
	{-0.044083, -0.024613, -0.117431, -0.058263,  0.008179,  0.011178, -0.041624, -0.123260},
	{-0.117735, -0.024345,  0.008927, -0.001303, -0.081491, -0.104572, -0.007043, -0.067123},
	{-0.112299, -0.007929, -0.052488, -0.116908, -0.030257,  0.009875,  0.000031, -0.080050},
	{ 0.005659, -0.038535, -0.118830, -0.041564, -0.003097, -0.109397, -0.074442, -0.001213},
	{-0.058675, -0.117962, -0.014533, -0.012484, -0.117591, -0.062163,  0.000398,  0.002208},
	{ 0.018759, -0.114775, -0.072420, -0.015796, -0.024858, -0.096423, -0.092574,  0.055902},
	{ 0.186206,  0.066666, -0.093901, -0.083400, -0.050103, -0.073462, -0.098693,  0.033705},
	{-0.080118, -0.054913, -0.083459, -0.089374,  0.073523,  0.196686,  0.042820, -0.099633},
	{ 0.054765,  0.196021,  0.053990, -0.096637, -0.079803, -0.052328, -0.079579, -0.097683},
	{-0.092917,  0.065871,  0.209082,  0.069072, -0.091233, -0.083159, -0.057583, -0.084435},
	{-0.080172, -0.094335,  0.051800,  0.212371,  0.093687, -0.080371, -0.086343, -0.058694},
	{-0.098998, -0.072622, -0.059005, -0.090925, -0.063341,  0.128669,  0.208913,  0.022446},
	{-0.060042, -0.079222, -0.091489,  0.056837,  0.222364,  0.105811, -0.074728, -0.087321},
	{ 0.017240, -0.093537, -0.070377, -0.061221, -0.093369, -0.052045,  0.145020,  0.205317},
	{ 0.286286,  0.081896, -0.036612, -0.052630, -0.051398, -0.018586,  0.133449,  0.321045},
	{-0.022913,  0.104241,  0.296008,  0.312626,  0.130534, -0.012370, -0.046058, -0.046925},
	{ 0.125102,  0.308023,  0.301202,  0.114413, -0.017407, -0.044594, -0.042965, -0.011909},
	{ 0.326466,  0.292408,  0.103527, -0.017697, -0.041094, -0.038607,  0.005764,  0.158558},
	{-0.041630, -0.020604,  0.094234,  0.286191,  0.333461,  0.167671,  0.008501, -0.038692},
	{ 0.090092, -0.015961, -0.038161, -0.032698,  0.027051,  0.193491,  0.340642,  0.274792},
	{-0.027770, -0.037982, -0.026487,  0.060731,  0.246749,  0.348872,  0.225464,  0.046336}
},
// 0.125 px/fr
{	{-0.000000,  0.897296,  0.353176,  0.000000,  0.000000,  1.209524,  0.285543,  0.265591},
	{ 1.029417,  0.000000,  0.000000,  0.000000,  0.620836,  0.000000,  0.188835,  0.246830},
	{-0.108047, -0.000000, -0.000000,  0.929848, -0.197093, -0.000000,  0.000000,  0.508013},
	{ 0.000000,  0.000000,  0.000000,  0.367456,  0.197863,  0.000000, -0.000000,  0.859015},
	{ 0.000000,  0.000000,  1.229100,  0.000000,  0.000000,  0.000000,  0.738794,  0.271190},
	{ 0.000000, -0.410008, -0.247282, -0.086121,  0.462063, -0.271767, -0.182609, -0.182525},
	{ 0.000000, -0.263183, -0.099207, -0.088605,  0.000000, -0.000000,  0.174004, -0.096171},
	{ 0.000000,  0.278772, -0.000000, -0.140555, -0.146193, -0.000000, -0.000000, -0.109096},
	{-0.201618, -0.000000,  0.000000, -0.193351, -0.268166, -0.162138,  0.555250, -0.276805},
	{-0.151171,  0.360803, -0.466397, -0.178297, -0.186825, -0.000000, -0.475992, -0.326441},
	{-0.000000, -0.000000, -0.000000, -0.277033,  0.374329, -0.000000, -0.210372, -0.264749},
	{-0.000000, -0.000000,  0.000000,  0.000000, -0.000000, -0.000000, -0.180506, -0.239941},
	{-0.395916, -0.000000, -0.195059, -0.224185, -0.413778, -0.191141, -0.156726, -0.000000},
	{-0.147002, -0.285356, -0.156458, -0.103351, -0.243213, -0.128499, -0.195833, -0.280861},
	{-0.189982, -0.737936, -0.455772, -0.300128, -0.382581, -0.523640, -0.524815, -0.397732},
	{ 0.000000,  0.000000,  0.287464,  0.000000,  0.200886,  0.000000,  0.338290,  0.285218},
	{-0.101822, -0.298001, -0.479286, -0.185336, -0.174942, -0.190061, -0.451103, -0.143887},
	{ 0.000000, -0.000000, -0.000000, -0.190980, -0.000000, -0.000000, -0.000000, -0.230796},
	{ 0.000000,  0.293190,  0.000000,  0.000000,  0.172343,  0.000000,  0.000000,  0.210156},
	{-0.000000,  0.430281,  0.305841,  0.200276,  0.000000,  0.000000,  0.363526,  0.321661},
	{ 0.000000, -0.000000, -0.108791, -0.143990,  0.000000,  0.000000, -0.145709, -0.197730},
	{ 0.000000,  0.204758,  0.000000,  0.000000,  0.200794,  0.000000,  0.271457,  0.000000},
	{-0.000000,  0.332910,  0.286988,  0.000000,  0.155198,  0.000000,  0.329061,  0.300256},
	{-0.000000, -0.165435, -0.000000, -0.092666, -0.128557, -0.000000, -0.000000, -0.269069},
	{-0.097398, -0.000000, -0.208955, -0.130879, -0.082892, -0.000000, -0.212524, -0.000000},
	{-0.105448, -0.491387, -0.410388, -0.190047, -0.237196, -0.307983, -0.477275, -0.285832},
	{-0.218714, -0.380534, -0.261717, -0.160753, -0.338830, -0.255540, -0.277978, -0.161782},
	{ 0.000000,  0.364896,  0.000000,  0.000000,  0.240844,  0.000000,  0.297409,  0.000000}
},
// 9 px/fr
{	{-4.864834, -4.867060, -4.823441, -4.740868, -4.646694, -4.603636, -4.662763, -4.786739},
	{-3.428012, -3.488151, -3.550758, -3.560310, -3.517467, -3.463406, -3.420058, -3.404072},
	{-0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000},
	{-1.957444, -2.017401, -2.055229, -2.057289, -2.021035, -1.947560, -1.893333, -1.904727},
	{-3.979133, -3.925736, -3.877434, -3.860755, -3.871451, -3.888292, -3.926100, -3.978706},
	{ 1.948717,  1.963352,  2.010421,  2.063527,  2.077270,  2.045093,  1.995961,  1.960698},
	{ 1.629890,  1.580667,  1.557382,  1.570485,  1.611004,  1.649102,  1.673758,  1.672242},
	{ 1.784991,  1.784529,  1.721898,  1.625747,  1.555419,  1.550628,  1.617398,  1.718844},
	{ 2.012012,  1.975361,  1.945907,  1.935350,  1.941052,  1.955430,  1.983538,  2.016023},
	{ 3.419318,  3.451937,  3.429333,  3.357931,  3.269283,  3.215087,  3.236822,  3.329197},
	{ 1.741699,  1.776702,  1.808409,  1.802087,  1.766176,  1.738879,  1.725429,  1.724178},
	{ 1.588804,  1.642456,  1.666208,  1.648262,  1.603281,  1.552858,  1.526549,  1.541457},
	{ 3.138541,  3.164963,  3.161345,  3.130037,  3.093148,  3.071225,  3.073344,  3.100179},
	{ 0.000000,  0.000000,  1.099349,  1.180536,  1.181763,  1.126990,  0.000000,  0.000000},
	{ 5.539593,  5.543994,  5.485430,  5.362374,  5.208172,  5.140021,  5.240953,  5.428928},
	{-4.056137, -4.117032, -4.056287, -3.905148, -3.762848, -3.703982, -3.756792, -3.904550},
	{ 2.270790,  2.128664,  2.040068,  2.067253,  2.168129,  2.257849,  2.320046,  2.343929},
	{-0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000},
	{-3.781555, -3.698213, -3.649388, -3.646792, -3.687281, -3.747849, -3.813056, -3.839367},
	{-4.309134, -4.343614, -4.415685, -4.477585, -4.459957, -4.384120, -4.318594, -4.299292},
	{-0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000},
	{-3.010623, -3.110226, -3.151800, -3.137590, -3.097021, -3.040314, -2.977231, -2.948042},
	{-2.954503, -2.839443, -2.696440, -2.662777, -2.755236, -2.858274, -2.933426, -2.978990},
	{ 1.209452,  1.377843,  1.404586,  1.263931,  1.109909,  1.029849,  0.000000,  0.000000},
	{ 2.089420,  2.032800,  1.842197,  1.695787,  1.641550,  1.658555,  1.762882,  1.948551},
	{ 4.438072,  4.492991,  4.604519,  4.721339,  4.695153,  4.525393,  4.405185,  4.402908},
	{ 4.205318,  4.047975,  3.943128,  3.896789,  3.932025,  4.088749,  4.284825,  4.331347},
	{-3.438845, -3.446991, -3.378377, -3.223788, -3.010716, -2.916123, -3.067748, -3.308302}
}};

// this corresponds to the number of spatiotemporal scales used
// same as pars.nScales in original S&H matlab code
// use only 1 spatiotemporal scale
//#define nrScales_ 3

// this filter is used to blur and downsample a 3D matrix
// same as filt in original S&H matlab code (see function blurDn3.m)
// blurring and downsampling is only used if nrScales_>1, that is, if we're processing at more than one
// spatiotemporal scale
#define scalingFiltSize 5
static const float h_scalingFilt[scalingFiltSize] = {0.0884, 0.3536, 0.5303, 0.3536, 0.0884};

// d_v1GaussFilt defines the 1D receptive field size of a V1 unit, which is then used for all three dimensions (X,Y and T)
// this guy can be reproduced in matlab with g=normpdf(-4:4,0,1.25);
// same as in original S&H matlab code
#define v1GaussFiltSize 9
static const float h_v1GaussFilt[v1GaussFiltSize] = {0.0007, 0.0155, 0.0903, 0.2345, 0.3179, 0.2345, 0.0903, 0.0155, 0.0007};

// d_complexV1Filt is the spacial filter for complex cells; it averages over "simple" V1 cells
// all simple cells must have the same space-time orientation and phase
// this guy can be reproduced in matlab with g=normpdf(-5:5,0,1.6);
// same as in original S&H matlab code
#define complexV1FiltSize 11
static const float h_complexV1Filt[complexV1FiltSize] = {0.0019, 0.0110, 0.0430, 0.1142, 0.2052, 0.2495, 0.2052, 0.1142, 0.0430, 0.0110, 0.0019};

// d_normV1filt is the spatial filter used complex cell normalization
// this guy can be reproduced in matlab with: g=normpdf(-10:10,0,3.35);
// same as in original S&H matlab code
// #define normV1filtSize 21
// __constant__ float d_normV1filt[normV1filtSize] = {0.0013, 0.0031, 0.0067, 0.0132, 0.0237, 0.0389, 0.0584, 0.0800, 0.1001, 0.1146, 0.1199, 0.1146, 0.1001, 0.0800, 0.0584, 0.0389, 0.0237, 0.0132, 0.0067, 0.0031, 0.0013};
//float* normV1filt;

// use a slightly bigger filter: g=normpdf(-12:12,0,5.0)/sum(g);
#define normV1filtSize 25
static const float h_normV1filt[normV1filtSize]={0.0045,0.0072,0.0109,0.0160,0.0225,0.0303,0.0393,0.0490,0.0587,0.0675,0.0746,0.0792,0.0808,0.0792,0.0746,0.0675,0.0587,0.0490,0.0393,0.0303,0.0225,0.0160,0.0109,0.0072,0.0045};

// difference operator for taking the first-order derivative
#define diff1filtSize 3
static const float h_diff1filt[diff1filtSize] = {-1/2.0, 0, 1/2.0};

// difference operator for taking the second-order derivative
#define diff2filtSize 3
static const float h_diff2filt[diff2filtSize] = {1, -2, 1};

// difference operator for taking the third-order derivative
// the grayscale values of our input stimuli will be convolved with d_scalingFilt in 3D
#define diff3filtSize 5
static const float h_diff3filt[diff3filtSize] = {-1/2.0, 1, 0, -1, 1/2.0};

// number of time steps to be considered in computation
// in the easiest case (only 1 spatiotemporal scale), it should be the same as v1GaussFiltSize
#define nrT 9


/// **************************************************************************************************************** ///
/// DEVICE FUNCTIONS
/// **************************************************************************************************************** ///

__global__ void dev_accumDiffStims(float *d_resp_tmp, float *diffV1GausBuf, int nrXnrY, int scale, int orderX, int orderY, int orderT) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	__shared__ float dirorders[nrFilters];

	if (threadIdx.x < nrFilters) {
		const float dir1 = d_v1popDirs[0][threadIdx.x]; // x-component
		const float dir2 = d_v1popDirs[1][threadIdx.x]; // y-component
		const float dir3 = d_v1popDirs[2][threadIdx.x]; // t-component

		float dirX = (orderX==0)?1:(orderX==1)?dir1:(orderX==2)?dir1*dir1:dir1*dir1*dir1;
		float dirY = (orderY==0)?1:(orderY==1)?dir2:(orderY==2)?dir2*dir2:dir2*dir2*dir2;
		float dirT = (orderT==0)?1:(orderT==1)?dir3:(orderT==2)?dir3*dir3:dir3*dir3*dir3;
		dirorders[threadIdx.x] = dirX*dirY*dirT;
	}

	__syncthreads();

	// TODO: find the fastest way to do this
	for(int i = tid; i < nrXnrY; i += threadN)
	{
//		int ii = i%nrXnrY;
//		int jj = i/nrXnrY;
		float d = diffV1GausBuf[i];

//		d_resp_[ii+jj*nrXnrY] += scale*d*dirorders[jj];

		for (int j=0; j<nrFilters; j++)
			d_resp_tmp[i+j*nrXnrY] += scale*d*dirorders[j];
	}
}

// parallel averaging
__global__ void dev_ave(float *i1data, float *i2data, float* odata, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		odata[i] = (i1data[i] + i2data[i])/2;
	}
}

// convolve idata with filt and store output in odata
__global__ void dev_conv1(float* idata, float* odata, int len, const float* filt, int filtlen) {
	__shared__ float block[CONV1_THREAD_SIZE];

	const int nrValidConv = CONV1_THREAD_SIZE - (filtlen-1);
	const int offset = (filtlen-1)/2;

	int xInd = blockIdx.x*nrValidConv + threadIdx.x - offset;
	int idx = blockIdx.y * len + xInd;

	block[threadIdx.x] = (xInd>=0 && xInd<len)?idata[idx]:0;

	__syncthreads();

	xInd += offset;
	idx += offset;

	if (xInd<len && threadIdx.x < nrValidConv) {
		float sum = 0;
		for (int i = 0; i< filtlen; i++)
			sum += block[threadIdx.x+i]*filt[i];
		odata[idx] = sum;
	}
}

__global__ void dev_convn(float* idata, float* odata, int nrX_, int nrN, int stride, int blockStride, int nrBlocks, const float* filt, int filtlen) {
	__shared__ float block[CONVN_THREAD_SIZE1*CONVN_THREAD_SIZE2];

	const int nrValidConv = (CONVN_THREAD_SIZE2-(filtlen-1));
	const int offset = (filtlen-1)/2;

	const int blockY = blockIdx.y/nrBlocks;
	const int b = blockIdx.y - blockY*nrBlocks;

	const int ind1 = blockIdx.x*CONVN_THREAD_SIZE1 + threadIdx.x;
	int ind2 = blockY*nrValidConv + threadIdx.y - offset;
	int idx = ind2*stride + ind1 + b*blockStride;

	const int threadxy = threadIdx.x*CONVN_THREAD_SIZE2+threadIdx.y;

	block[threadxy] = (ind2>=0 && ind2<nrN && ind1 < nrX_)?idata[idx]:0;

	__syncthreads();

	ind2 += offset;
	idx += offset*stride;

	if (ind2<nrN && ind1 < nrX_ && threadIdx.y < nrValidConv) {
		float sum = 0;
		for (int i = 0; i< filtlen; i++) sum += block[threadxy+i]*filt[i];
		odata[idx] = sum;
	}
}

__global__ void dev_convRowsGeneral(float* idata, float* odata, int nrX_, int nrY_, int nrZ_, const float* filt, int filtlen) {
	const int tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);
	const int total = nrX_ * nrY_ * nrZ_;
	const int offset = (filtlen - 1) / 2;

	for (int i = tid; i < total; i += threadN) {
		int x = i % nrX_;
		int y = (i / nrX_) % nrY_;
		int z = i / (nrX_ * nrY_);
		float sum = 0.0f;
		for (int k = 0; k < filtlen; k++) {
			int xx = x + k - offset;
			if (xx >= 0 && xx < nrX_) {
				sum += idata[xx + y * nrX_ + z * nrX_ * nrY_] * filt[k];
			}
		}
		odata[i] = sum;
	}
}

__global__ void dev_convColsGeneral(float* idata, float* odata, int nrX_, int nrY_, int nrZ_, const float* filt, int filtlen) {
	const int tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);
	const int total = nrX_ * nrY_ * nrZ_;
	const int offset = (filtlen - 1) / 2;

	for (int i = tid; i < total; i += threadN) {
		int x = i % nrX_;
		int y = (i / nrX_) % nrY_;
		int z = i / (nrX_ * nrY_);
		float sum = 0.0f;
		for (int k = 0; k < filtlen; k++) {
			int yy = y + k - offset;
			if (yy >= 0 && yy < nrY_) {
				sum += idata[x + yy * nrX_ + z * nrX_ * nrY_] * filt[k];
			}
		}
		odata[i] = sum;
	}
}

// consider edge effects
__global__ void dev_edges(float *data, int len, int nrX_, int nrY_) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	float edgeFactor;

	for(int i = tid; i < len; i += threadN) {
		int X = i%nrX_;
		int Y = (i/nrX_)%nrY_;
		int scale = i/(nrX_*nrY_*28);

		// we decrease filter responses near edges (use a different scaling factor per spatiotemporal scale level)
		// scaling factors are chosen such that no more edge effects are visible in the direction tuning task, whatever
		// works...
		float edgedist = (float)min(min(X,nrX_-1-X),min(Y,nrY_-1-Y)); // box-shaped instead of round
		if (scale==0)
			edgeFactor = fmin(125.0f,edgedist*edgedist*edgedist)/125.0f; // 5px^3
		else if (scale==1)
			edgeFactor = fmin(1296.0f,edgedist*edgedist*edgedist*edgedist)/1296.0f; // 6px^4
		else
			edgeFactor = fmin(7776.0f,edgedist*edgedist*edgedist*edgedist*edgedist)/7776.0f; // 6px^5

		data[i] *= edgeFactor;

/*		if (X<nrX_/2)
			data[i] = 0;
		if (Y<nrY_/2)
			data[i] = 0; */
	}
}

// linearly combines filter responses with input activity to get direction-selective output activity
__global__ void dev_filt2dir(float* d_respIn, float* d_respOut, unsigned int len, int nrXnrY, int nrScales, double speed=1.0) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	// for now we store speeds [1,2,...,10] pixels/frame in an array
	// so make sure to map speeds back to the right array index
	// \TODO
	int speedInd;
	if (speed==1.5)
		speedInd = 0;
	else if (speed==0.125)
		speedInd = 1;
	else if (speed==9.0)
		speedInd = 2;
	else
		return;

	for (unsigned int i=tid; i<len; i+=threadN) {
		unsigned int ind = i%(nrXnrY*nrFilters*nrScales); // which index in d_respIn
		int spaceTimeInd = (i/nrXnrY)%nrFilters; // which filter
		int pos = i%nrXnrY; // which position in space
		int dir = i/(nrXnrY*nrFilters*nrScales); // which direction of d_respOut

		d_respOut[dir*nrXnrY+pos] += d_respIn[ind]*motionProj[speedInd][spaceTimeInd][dir];
	}
}

// parallel full-wave rectification
__global__ void dev_fullRect2(float *data, int len, double scaleV1Linear, double scaleV1FullWaveRect) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		// scaleV1Linear is to be applied before squaring (shModelV1Linear.m)
		float d = data[i]*scaleV1Linear;
//		d = (d>0)?d:0; // Matlab code and Rust et al, 2006 do not half-rectify anymore
		// scaleV1FullWaveRect is to be applied after squaring (shModelFullWaveRectification.m)
		data[i] = d*d*scaleV1FullWaveRect;
	}
}

// compute the mean on the array's third dimension
// this is used to compute the mean of all 28 filter responses at a given location/scale (used in the complex cell
// normalization step)
__global__ void dev_mean3(float *idata, float *odata, int nrXnrY, int nrZ) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);
	const int blockSize = nrXnrY*nrZ;

	for(int i = tid; i < nrXnrY; i += threadN) {
		float sum = 0;
		int ind = i + blockIdx.y*blockSize;

		for (int j=0; j < nrZ; j++)
			sum += idata[ind+j*nrXnrY];
		odata[i+blockIdx.y*nrXnrY] = sum/nrZ;
	}
}

// population normalization of complex cell responses
__global__ void dev_normalize(float *resp, float *pop, int nrXnrY, double scaleV1C50) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);
	const int blockSize = nrXnrY*nrFilters;

	for(int i = tid; i < nrXnrY; i += threadN) {
		float norm = pop[i+blockIdx.y*nrXnrY];
		int ind = i + blockIdx.y*blockSize;

		// TODO: this inner loop is unnecessary, parallelize
		for (int j=0; j < nrFilters; j++)
			resp[ind+j*nrXnrY] /= (norm + scaleV1C50*scaleV1C50); // scaleFactors.v1sigma
	}
}

// parallel mulitplying with a scale factor
__global__ void dev_scale(float *data, float scale, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		data[i] *= scale;
	}
}

// parallel half-squaring and scaling (with min == spontaneous activity)
__global__ void dev_scaleHalfRect(float *data, int len, float scale, float spont) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN)
		data[i] = (data[i]>0)?data[i]*scale:spont;
}

// reads in stimulus in grayscale format and scales to [0,1]
__global__ void dev_split_gray(unsigned char *idata, float *gray, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		gray[i] = idata[i]*0.00392156863; // mult is faster than 1/255
	}
}

// reads in stimuli in RGB format and extracts R, G, B, and grayscale values (normalized to [0,1])
__global__ void dev_split_RGB(unsigned char *idata, float *gray, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	typedef struct rgb_s {
		unsigned char r,g,b;
	} rgb_t;

	rgb_t* rgbs = (rgb_t*)idata;

	for(int i = tid; i < len; i += threadN) {
		rgb_t rgb=rgbs[i];
		// Weighted RGB luminance requested for the KITTI RGB frontend.
		// Multiplication by 1/255 keeps the result on the input intensity scale.
		gray[i] = (0.1299f * (float)rgb.r
		         + 0.5870f * (float)rgb.g
		         + 0.1140f * (float)rgb.b) * (1.0f / 255.0f);

	}
}

// parallel subtraction
__global__ void dev_sub(float *i1data, float *i2data, float* odata, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		odata[i] = i1data[i] - i2data[i];
	}
}

// parallel summing
__global__ void dev_sum(float *i1data, float *i2data, float* odata, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		odata[i] = i1data[i] + i2data[i];
	}
}


/// **************************************************************************************************************** ///
/// OMS DEVICE FUNCTIONS
/// **************************************************************************************************************** ///

// OMS: update eventSurface and activitySurface from ON/OFF frames
// eventSurface = decay * eventSurface + eventGain * (ON - OFF)
// activitySurface = decay * activitySurface + (ON + OFF)
__global__ void dev_eventUpdateKernel(
	unsigned char* d_onFrame,
	unsigned char* d_offFrame,
	float* d_eventSurface,
	float* d_activitySurface,
	int nrX, int nrY,
	float decay,
	float eventGain
) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);
	const int len = nrX * nrY;

	for (int i = tid; i < len; i += threadN) {
		float onVal = d_onFrame[i] * 0.00392156863f;  // /255
		float offVal = d_offFrame[i] * 0.00392156863f;

		float eventSignal = eventGain * (onVal - offVal);
		float eventActivity = onVal + offVal;

		d_eventSurface[i] = decay * d_eventSurface[i] + eventSignal;
		d_activitySurface[i] = decay * d_activitySurface[i] + eventActivity;
	}
}

// OMS: threshold positive DoG values relative to the global maximum.
__global__ void dev_omsThresholdPositive(
	float* d_dog,
	unsigned char* d_mask,
	float maxVal,
	float thresholdRel,
	int len
) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	float threshold = thresholdRel * fmaxf(maxVal, 0.0f);

	for (int i = tid; i < len; i += stride) {
		d_mask[i] = (d_dog[i] > 0.0f && d_dog[i] >= threshold) ? 255 : 0;
	}
}

// OMS: binary mask gate — pass eventSurface where mask > 0, else 0
__global__ void dev_omsThresholdPositiveOr(
	float* d_dog,
	unsigned char* d_scaleMask,
	unsigned char* d_finalMask,
	float maxVal,
	float thresholdRel,
	int len
) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	float threshold = thresholdRel * fmaxf(maxVal, 0.0f);

	for (int i = tid; i < len; i += stride) {
		unsigned char m = (d_dog[i] > 0.0f && d_dog[i] >= threshold) ? 255 : 0;
		d_scaleMask[i] = m;
		if (m > 0) d_finalMask[i] = 255;
	}
}

__global__ void dev_maxInPlace(float* dst, const float* src, int len) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	for (int i = tid; i < len; i += stride) {
		dst[i] = fmaxf(dst[i], src[i]);
	}
}

__global__ void dev_omsGateSoft(
	float* d_eventSurface,
	unsigned char* d_mask,
	float* d_stimBufLatest,
	float backgroundKeep,
	int len
) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	for (int i = tid; i < len; i += stride) {
		float keep = (d_mask[i] > 0) ? 1.0f : backgroundKeep;
		d_stimBufLatest[i] = d_eventSurface[i] * keep;
	}
}

// OMS: block-level min/max reduction (first pass)
__global__ void dev_reduceMinMax(
	float* data,
	float* blockMin, float* blockMax,
	int len, int* d_nBlocks
) {
	extern __shared__ float shared[];
	float* sMin = shared;
	float* sMax = &shared[blockDim.x];

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	float lmin = 1e30f, lmax = -1e30f;
	for (int i = bid * blockDim.x + tid; i < len; i += stride) {
		lmin = fminf(lmin, data[i]);
		lmax = fmaxf(lmax, data[i]);
	}

	sMin[tid] = lmin;
	sMax[tid] = lmax;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sMin[tid] = fminf(sMin[tid], sMin[tid + s]);
			sMax[tid] = fmaxf(sMax[tid], sMax[tid + s]);
		}
		__syncthreads();
	}

	if (tid == 0) {
		blockMin[bid] = sMin[0];
		blockMax[bid] = sMax[0];
		if (bid == 0) *d_nBlocks = gridDim.x;
	}
}

// OMS: final min/max reduction (second pass, 1 block)
__global__ void dev_finalMinMax(
	float* blockMin, float* blockMax,
	float* outMin, float* outMax,
	int nBlocks
) {
	extern __shared__ float shared[];
	float* sMin = shared;
	float* sMax = &shared[blockDim.x];

	int tid = threadIdx.x;
	float lmin = 1e30f, lmax = -1e30f;
	for (int i = tid; i < nBlocks; i += blockDim.x) {
		lmin = fminf(lmin, blockMin[i]);
		lmax = fmaxf(lmax, blockMax[i]);
	}

	sMin[tid] = lmin;
	sMax[tid] = lmax;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sMin[tid] = fminf(sMin[tid], sMin[tid + s]);
			sMax[tid] = fmaxf(sMax[tid], sMax[tid + s]);
		}
		__syncthreads();
	}

	if (tid == 0) {
		*outMin = sMin[0];
		*outMax = sMax[0];
	}
}


/// **************************************************************************************************************** ///
/// CONSTRUCTOR / DESTRUCTOR
/// **************************************************************************************************************** ///

MotionEnergy::MotionEnergy(int _nrX, int _nrY, int _nrC) {
	nrX_ = _nrX;
	nrY_ = _nrY;
	nrC_ = _nrC;
	assert(nrC_==1 || nrC_==3);

	// from shGetDims
	// \TODO compute instead of hardcoding (in case spatial filter sizes, etc., change)
	min_nrX_ = 19;
	min_nrY_ = 19;

	nrScales_ = 3;

	// shPars
	initParams();

	initME();
}

MotionEnergy::~MotionEnergy() {
	CUDA_CHECK_ERRORS(cudaFree(scalingFilt));
	CUDA_CHECK_ERRORS(cudaFree(v1Gaus));
	CUDA_CHECK_ERRORS(cudaFree(complexV1Filt));
	CUDA_CHECK_ERRORS(cudaFree(normV1filt));
	CUDA_CHECK_ERRORS(cudaFree(diff1filt));
	CUDA_CHECK_ERRORS(cudaFree(diff2filt));
	CUDA_CHECK_ERRORS(cudaFree(diff3filt));
	CUDA_CHECK_ERRORS(cudaFree(d_stimBuf));
	CUDA_CHECK_ERRORS(cudaFree(diffV1GausBufT));
	CUDA_CHECK_ERRORS(cudaFree(d_stim));
	CUDA_CHECK_ERRORS(cudaFree(d_scalingStimBuf));
	CUDA_CHECK_ERRORS(cudaFree(d_v1GausBuf));
	CUDA_CHECK_ERRORS(cudaFree(d_diffV1GausBuf));
	CUDA_CHECK_ERRORS(cudaFree(d_pop));
	CUDA_CHECK_ERRORS(cudaFree(d_resp_));
	CUDA_CHECK_ERRORS(cudaFree(d_respV1c));
	CUDA_CHECK_ERRORS(cudaFree(d_eventSurface));
	CUDA_CHECK_ERRORS(cudaFree(d_activitySurface));
	CUDA_CHECK_ERRORS(cudaFree(d_omsCenter));
	CUDA_CHECK_ERRORS(cudaFree(d_omsSurround));
	CUDA_CHECK_ERRORS(cudaFree(d_omsDogDebug));
	CUDA_CHECK_ERRORS(cudaFree(d_omsMask));
	CUDA_CHECK_ERRORS(cudaFree(d_omsMaskScale));
	CUDA_CHECK_ERRORS(cudaFree(d_onFrame));
	CUDA_CHECK_ERRORS(cudaFree(d_offFrame));
	// OMS Gaussian + workspace buffers
	CUDA_CHECK_ERRORS(cudaFree(d_centerGauss_));
	CUDA_CHECK_ERRORS(cudaFree(d_surroundGauss_));
	for (int s = 0; s < 3; s++) {
		CUDA_CHECK_ERRORS(cudaFree(d_multiCenterGauss_[s]));
		CUDA_CHECK_ERRORS(cudaFree(d_multiSurroundGauss_[s]));
	}
	CUDA_CHECK_ERRORS(cudaFree(d_omsTmp_));
	CUDA_CHECK_ERRORS(cudaFree(d_minBlock_));
	CUDA_CHECK_ERRORS(cudaFree(d_maxBlock_));
	CUDA_CHECK_ERRORS(cudaFree(d_nBlocks_));
}


/// **************************************************************************************************************** ///
/// PUBLIC METHODS
/// **************************************************************************************************************** ///

// sets the number of channels (grayscale or rgb)
void MotionEnergy::setNumChannels(int nrC) {
	assert(nrC==1 || nrC==3);

	// if number of channels has changed, re-allocate
	if (nrC_ != nrC) {
		nrC_ = nrC;
		CUDA_CHECK_ERRORS(cudaFree(d_stim));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_stim, nrX_*nrY_*nrC_));
	}
}


void MotionEnergy::calcV1complex(unsigned char* stim, float* V1comp, double speed, bool GPUpointers) {
	// for now, only these hardcoded speeds are supported
	assert(speed==1.5 || speed==0.125 || speed==9.0);

	// allocate stim on device
	loadInput(stim);

	// convolve the stimulus with separate V1 filters
	calcV1linear();

	// rectify linear responses to get V1 simple cell firing rate
	calcV1rect();

	// spatial pooling to get V1 complex
	// calcV1blur();

	// divisive normalization
	calcV1normalize();

	// init open/ new close
	// steer filters in specified directions
	// calcV1direction(speed);

	// 8
	// copy V1 complex responses to output array
	// CUDA_CHECK_ERRORS(cudaMemcpy(V1comp, d_respV1c, sizeof(float)*nrX_*nrY_*nrDirs,
	// 	GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

	// 28
	CUDA_CHECK_ERRORS(cudaMemcpy(V1comp, d_resp_, sizeof(float)*nrX_*nrY_*nrFilters*nrScales_,
		GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

}




// event-based entry point with OMS gating
// receives ON/OFF event frames instead of RGB/grayscale frames
void MotionEnergy::calcV1complexEvents(
	unsigned char* onEvents,
	unsigned char* offEvents,
	float* V1comp,
	double speed,
	bool GPUpointers
) {
	assert(speed==1.5 || speed==0.125 || speed==9.0);

	// 1. copy ON/OFF event frames to GPU
	CUDA_CHECK_ERRORS(cudaMemcpy(d_onFrame, onEvents, nrX_*nrY_, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMemcpy(d_offFrame, offEvents, nrX_*nrY_, cudaMemcpyHostToDevice));

	// 2. shift d_stimBuf in time by 1 frame (oldest out, newest slot freed)
	for (int i = 1; i < nrT; i++)
		CUDA_CHECK_ERRORS(cudaMemcpy(&d_stimBuf[nrX_*nrY_*(i-1)], &d_stimBuf[nrX_*nrY_*i],
			sizeof(float)*nrX_*nrY_, cudaMemcpyDeviceToDevice));

	// 3. update event surfaces on GPU from ON/OFF input
	//    eventSurface = decay * eventSurface + eventGain * (ON - OFF)
	//    activitySurface = decay * activitySurface + (ON + OFF)
	{
		int gridSize = iDivUp(nrX_*nrY_, 256);
		dev_eventUpdateKernel<<<gridSize, 256>>>(
			d_onFrame, d_offFrame,
			d_eventSurface, d_activitySurface,
			nrX_, nrY_, eventDecay_, eventGain_
		);
		CUDA_GET_LAST_ERROR("dev_eventUpdateKernel() execution failed\n");
	}

	// 4. Compute OMS independently from the signed event surface.  Legacy
	// pre-gating remains available, while mask-only keeps the raw event surface
	// for V1/MT and exports OMS solely as an object-motion cue.
	if (omsMode_ == OMS_OFF) {
		CUDA_CHECK_ERRORS(cudaMemset(d_omsMask, 0, sizeof(unsigned char)*nrX_*nrY_));
		CUDA_CHECK_ERRORS(cudaMemset(d_omsDogDebug, 0, sizeof(float)*nrX_*nrY_));
		CUDA_CHECK_ERRORS(cudaMemcpy(&d_stimBuf[nrX_*nrY_*(nrT-1)], d_eventSurface,
			sizeof(float)*nrX_*nrY_, cudaMemcpyDeviceToDevice));
	} else {
		applyOMS(omsMode_ == OMS_PRE_GATE);
	}
	if (omsDebug_ && omsDebugInterval_ > 0 && (omsDebugFrame_ % omsDebugInterval_) == 0) {
		dumpOmsDebugFrame();
	}
	omsDebugFrame_++;

	// 5. copy d_stimBuf to d_scalingStimBuf (preserves time structure for V1 pipeline)
	CUDA_CHECK_ERRORS(cudaMemcpy(d_scalingStimBuf, d_stimBuf, sizeof(float)*nrX_*nrY_*nrT,
		cudaMemcpyDeviceToDevice));

	// 6. reset response buffers
	CUDA_CHECK_ERRORS(cudaMemset(d_resp_, 0, sizeof(float)*nrX_*nrY_*nrFilters*nrScales_));
	CUDA_CHECK_ERRORS(cudaMemset(d_respV1c, 0, sizeof(float)*nrX_*nrY_*nrDirs));

	// 7. standard V1 pipeline (unchanged from RGB/grayscale path)
	calcV1linear();
	calcV1rect();
	calcV1blur();
	calcV1normalize();
	// init open/ new close
	// calcV1direction(speed);

	// 8. copy V1 filter responses to output array
	// 28 dir
	CUDA_CHECK_ERRORS(cudaMemcpy(V1comp, d_resp_, sizeof(float)*nrX_*nrY_*nrFilters*nrScales_,
		GPUpointers ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost));

	// 8 dir
	// CUDA_CHECK_ERRORS(cudaMemcpy(V1comp, d_respV1c, sizeof(float)*nrX_*nrY_*nrDirs,
	// 	GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
}
// allocates a frame of the input stim on device
void MotionEnergy::loadInput(unsigned char* stim) {
	// load input
	CUDA_CHECK_ERRORS(cudaMemcpy(d_stim,stim,nrC_*nrX_*nrY_,cudaMemcpyHostToDevice));

	// shift d_stimBuf in time by 1 frame, from frame i to frame i-1
	for(int i=1;i<nrT;i++)
		CUDA_CHECK_ERRORS(cudaMemcpy(&d_stimBuf[nrX_*nrY_*(i-1)],&d_stimBuf[nrX_*nrY_*i],sizeof(float)*nrX_*nrY_,//
			cudaMemcpyDeviceToDevice));

	if (nrC_==3) {
		// RGB input
		dev_split_RGB<<<iDivUp(nrX_*nrY_,128), 128>>>(d_stim, &d_stimBuf[nrX_*nrY_*(nrT-1)], nrX_*nrY_);// load data into end
	} else {
		// GRAY input
		dev_split_gray<<<iDivUp(nrX_*nrY_,128), 128>>>(d_stim, &d_stimBuf[nrX_*nrY_*(nrT-1)], nrX_*nrY_);
	}
	CUDA_GET_LAST_ERROR("dev_split() execution failed\n");


	// ME responses do not depend on color responses, but on grayscale values found in d_stimBuf[]



	// allocate d_resp_, which will contain the response to all 28 (nrFilters) space-time orientation at 3 (nrScales_)
	// scales for every pixel location (x,y)
	CUDA_CHECK_ERRORS(cudaMemset (d_resp_, 0, sizeof(float)*nrX_*nrY_*nrFilters*nrScales_));

	// working copy of grayscale values: copy d_stimBuf to d_scalingStimBuf
	CUDA_CHECK_ERRORS(cudaMemcpy(d_scalingStimBuf,d_stimBuf,sizeof(float)*nrX_*nrY_*nrT,cudaMemcpyDeviceToDevice));

	// reset V1complex responses to 0
	CUDA_CHECK_ERRORS(cudaMemset(d_respV1c, 0, sizeof(float)*nrX_*nrY_*nrDirs));
}

void MotionEnergy::calcV1linear() {
	// compute the V1 simple cell responses at different spatial scales
	// the i-th stage blurs and downsamples the image (i-1) times
	for (int scale=1; scale<=nrScales_; scale++) {
		// blur/scale the image... each time this is called stim is blurred more
		// scale 1 == original image resolution (space/time)
		// printf("changed 603\n");
		if (scale > 4) {
			float* tmp;
			CUDA_CHECK_ERRORS(cudaMalloc((void**)&tmp, sizeof(float)*nrX_*nrY_*nrT));

			// convolve d_scalingStimBuf by scalingFilt in 3D
			uint3 sizes = make_uint3(nrX_,nrY_,nrT);
			conv3D(d_scalingStimBuf, tmp, sizes, scalingFilt, scalingFiltSize);

			CUDA_CHECK_ERRORS(cudaFree(d_scalingStimBuf));
			d_scalingStimBuf = tmp;
		}

		// nrT is 9, v1GaussSize is 9, so we're taking d_scalingStimBuf[0-0+nrX_*nrY_*9]
		// since nrT could be greater than v1GaussSize, we take "only the part we want", quote Micah comment
		CUDA_CHECK_ERRORS(cudaMemcpy(d_v1GausBuf, &d_scalingStimBuf[nrX_*nrY_*((nrT-v1GaussFiltSize)/2)],
			sizeof(float)*nrX_*nrY_*v1GaussFiltSize, cudaMemcpyDeviceToDevice));

		float* tmp;
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&tmp, sizeof(float)*nrX_*nrY_*v1GaussFiltSize));

		// convolve d_v1GausBuf by v1Gaus in 3D
		uint3 sizes = make_uint3(nrX_,nrY_,v1GaussFiltSize);
		conv3D(d_v1GausBuf, tmp, sizes, v1Gaus, v1GaussFiltSize);
		CUDA_CHECK_ERRORS(cudaFree(d_v1GausBuf));
		d_v1GausBuf = tmp;

		// go through and calculate all directional derivatives and then combine them to calculate the diferent
		// space-time oriented filters
		for (int orderT=0; orderT<=3; orderT++) {
			// reset diffV1GausBufT back to the 3D gaussian filtered version
			CUDA_CHECK_ERRORS(cudaMemcpy(diffV1GausBufT, d_v1GausBuf, sizeof(float)*nrX_*nrY_*v1GaussFiltSize,
				cudaMemcpyDeviceToDevice));

			if (orderT > 0) {
				// take the derivative
				// sizes: tripel (nrX_,nrY_,v1GaussSize)
				diffV1GausBufT = diff(diffV1GausBufT, sizes, orderT,2);
			}

			for (int orderY=0; orderY<=3-orderT; orderY++) {
				int orderX = 3-orderY-orderT;

				CUDA_CHECK_ERRORS(cudaMemcpy(d_diffV1GausBuf, diffV1GausBufT, sizeof(float)*nrX_*nrY_*v1GaussFiltSize,
					cudaMemcpyDeviceToDevice));

				if (orderX > 0) d_diffV1GausBuf = diff(d_diffV1GausBuf, sizes, orderX,0);
				if (orderY > 0) d_diffV1GausBuf = diff(d_diffV1GausBuf, sizes, orderY,1);

				// combine the directional derivative by the direction of the space-time filter
				// this is basically doing what shSwts.m did in the original S&H matlab code
				accumDiffStims(&d_resp_[(scale-1)*nrX_*nrY_*nrFilters], &d_diffV1GausBuf[nrX_*nrY_*(v1GaussFiltSize/2)],
					sizes, orderX, orderY, orderT);
			}
		}
	}

	// Note: the scaling factor scaleV1Linear will be applied in calcV1rect()

	// consider edge effects
	dev_edges<<<iDivUp(nrX_*nrY_*nrFilters*nrScales_,128), 128>>>(d_resp_, nrX_*nrY_*nrFilters*nrScales_, nrX_, nrY_);
	CUDA_GET_LAST_ERROR("dev_edges() execution failed\n");
}


void MotionEnergy::calcV1rect() {
	// full-wave rectification of linear responses
	// contains scaleFactor.v1Linear and scaleFactor.v1FullWaveRectified

	int len = nrX_*nrY_*nrFilters*nrScales_;
	dev_fullRect2<<<iDivUp(nrX_*nrY_*nrFilters*nrScales_,128), 128>>>(d_resp_, len, scaleV1Linear_,
		scaleV1FullWaveRect_);//

	CUDA_GET_LAST_ERROR("dev_fullRect2() execution failed\n");
}



void MotionEnergy::calcV1blur() {
	float* tmp;

	// complex: convolve by d_complexV1Filt in 2D
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&tmp, sizeof(float)*nrX_*nrY_*nrFilters*nrScales_));
	uint3 sizes = make_uint3(nrX_,nrY_,nrFilters*nrScales_);
	conv2D(d_resp_, tmp, sizes, complexV1Filt, complexV1FiltSize);
	CUDA_CHECK_ERRORS(cudaFree(tmp));

	// scale with scaleFactors.v1Blur
	// NOTE: scaling with 1.0205..? Skip to save computation time
	dev_scale<<<iDivUp(nrX_*nrY_*nrFilters*nrScales_,128), 128>>>(d_resp_, scaleV1Blur_, nrX_*nrY_*nrFilters*nrScales_);
	CUDA_GET_LAST_ERROR("dev_scale() execution failed\n");
}


void MotionEnergy::calcV1normalize() {
	float* tmp;

	// we need to associate each filter at pixel position (x,y) with a power/intensity, but there are 28 filter
	// responses at each location... so we need to (i) average over the 28 filters (3rd dimension in d_resp_) and put it
	// into d_pop ...对28个过滤器取平均值
	dim3 gridm(iDivUp(nrX_*nrY_,128), nrScales_);
	dev_mean3<<<gridm, 128>>>(d_resp_, d_pop, nrX_*nrY_, nrFilters);
	CUDA_GET_LAST_ERROR("dev_mean3() execution failed\n");

	// ... (ii) scale with scaleFactors.v1Complex
	// NOTE: Scale with 0.99..? Skip to save computation time
	dev_scale<<<iDivUp(nrX_*nrY_*nrFilters*nrScales_,128), 128>>>(d_resp_, scaleV1Complex_,
		nrX_*nrY_*nrFilters*nrScales_);
	CUDA_GET_LAST_ERROR("dev_scale() execution failed\n");


	// ... and (iii) sum over some spatial neighborhood 对空间领域求和
	// population normalization: convolve by d_normV1filtSize in 2D
	uint3 nsizes = make_uint3(nrX_,nrY_,nrScales_);
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&tmp, sizeof(float)*nrX_*nrY_*nrScales_));
	conv2D(d_pop, tmp, nsizes, normV1filt, normV1filtSize);
	CUDA_CHECK_ERRORS(cudaFree(tmp));

	// \TODO
	// don't scale with scaleFactors.v1NormalizationStrength * scaleFactors.v1NormalizationPopulationK
	// since we don't normalize over the WHOLE population, these factors are off
	// the purpose of this normalization is to get a htan()-like response normalization: a scaling factor of 1.0 turns
	// out to be good enough
	dev_scale<<<iDivUp(nrX_*nrY_*nrScales_,128), 128>>>(d_pop, scaleV1NormStrength_*scaleV1NormPopK_,
		nrX_*nrY_*nrScales_);
	CUDA_GET_LAST_ERROR("dev_scale() execution failed\n");
// 候选方案：补偿归一化参考中的空间权重损失
const bool enableNormCompensation = true;

if (enableNormCompensation) {
    const int pixels = nrX_ * nrY_;
    const int populationCount = pixels * nrScales_;

    std::vector<float> hostPop(populationCount);

    CUDA_CHECK_ERRORS(cudaMemcpy(
        hostPop.data(),
        d_pop,
        sizeof(float) * populationCount,
        cudaMemcpyDeviceToHost
    ));

    const int radius = (normV1filtSize - 1) / 2;

    double kernelSum = 0.0;
    for (int k = 0; k < normV1filtSize; k++) {
        kernelSum += h_normV1filt[k];
    }

    const double fullWeight = kernelSum * kernelSum;

    for (int s = 0; s < nrScales_; s++) {
        for (int y = 0; y < nrY_; y++) {
            for (int x = 0; x < nrX_; x++) {

                double effectiveWeight = 0.0;

                for (int ky = 0; ky < normV1filtSize; ky++) {
                    const int yy = y + ky - radius;
                    if (yy < 0 || yy >= nrY_) continue;

                    for (int kx = 0; kx < normV1filtSize; kx++) {
                        const int xx = x + kx - radius;
                        if (xx < 0 || xx >= nrX_) continue;

                        const int distance = std::min(
                            std::min(xx, nrX_ - 1 - xx),
                            std::min(yy, nrY_ - 1 - yy)
                        );

                        const double d = distance;
                        double edgeFactor;

                        if (s == 0) {
                            edgeFactor = std::min(
                                1.0, d*d*d / 125.0
                            );
                        }
                        else if (s == 1) {
                            edgeFactor = std::min(
                                1.0, d*d*d*d / 1296.0
                            );
                        }
                        else {
                            edgeFactor = std::min(
                                1.0, d*d*d*d*d / 7776.0
                            );
                        }

                        effectiveWeight +=
                            double(h_normV1filt[kx]) *
                            double(h_normV1filt[ky]) *
                            edgeFactor * edgeFactor;
                    }
                }

                if (effectiveWeight <= 0.0) {
                    throw std::runtime_error(
                        "Normalization compensation has zero support"
                    );
                }

                const int index = s * pixels + y * nrX_ + x;

                hostPop[index] = float(
                    hostPop[index] * fullWeight / effectiveWeight
                );
            }
        }
    }

    CUDA_CHECK_ERRORS(cudaMemcpy(
        d_pop,
        hostPop.data(),
        sizeof(float) * populationCount,
        cudaMemcpyHostToDevice
    ));
}
	// 调试：只记录本次MotionEnergy处理的第10帧
static int normDebugFrame = 0;
normDebugFrame++;

bool saveNormDebug = false;

// 把GPU浮点数组复制到CPU，再保存为二进制文件
auto saveDebugArray = [](const char* path,
                         const float* deviceData,
                         size_t count) {
    float* hostData = new float[count];

    cudaError_t status = cudaMemcpy(
        hostData,
        deviceData,
        count * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (status != cudaSuccess) {
        delete[] hostData;
        throw std::runtime_error(cudaGetErrorString(status));
    }

    FILE* file = fopen(path, "wb");

    if (file == NULL) {
        delete[] hostData;
        throw std::runtime_error("Cannot open normalization debug file");
    }

    size_t written = fwrite(hostData, sizeof(float), count, file);
    int closeStatus = fclose(file);
    delete[] hostData;

    if (written != count || closeStatus != 0) {
        throw std::runtime_error("Cannot finish normalization debug file");
    }
};

if (saveNormDebug) {
    size_t responseCount =
        size_t(nrX_) * nrY_ * nrFilters * nrScales_;

    size_t populationCount =
        size_t(nrX_) * nrY_ * nrScales_;

    saveDebugArray(
        "results/v1_norm_probe_01/before.bin",
        d_resp_,
        responseCount
    );

    saveDebugArray(
        "results/v1_norm_probe_01/pop.bin",
        d_pop,
        populationCount
    );

    printf("[NORM_DEBUG] frame=10, C50=%.9g\n", scaleV1C50_);
}

	// d_resp_ is the numerator, d_pop the denominator sum term
	dev_normalize<<<gridm, 128>>>(d_resp_, d_pop, nrX_*nrY_, scaleV1C50_);
	CUDA_GET_LAST_ERROR("dev_normalize() execution failed\n");

	if (saveNormDebug) {
    size_t responseCount =
        size_t(nrX_) * nrY_ * nrFilters * nrScales_;

    saveDebugArray(
        "results/v1_norm_probe_01/after.bin",
        d_resp_,
        responseCount
    );

    printf("[NORM_DEBUG] before, pop and after exported\n");
}

	//2 中内容
// 	for (int scale=0; scale<nrScales_; scale++)
// 		dev_scale<<<iDivUp(nrX_*nrY_*nrFilters,128), 128>>>(&d_resp_[scale*nrX_*nrY_*nrFilters], (scale==0?15.0f:(scale==1?17.0f:11.0f)), nrX_*nrY_*nrFilters); // 15 17.5 17
}


void MotionEnergy::calcV1direction(double speed) {
	// The 28 filter responses do now need to be collapsed onto the directions and speeds of motion specified in
	// motionProj1, motionProj2, motionProj3

	dev_filt2dir<<<iDivUp(nrX_*nrY_*nrFilters*nrScales_*nrDirs,256), 256>>>(d_resp_, d_respV1c,
		nrX_*nrY_*nrFilters*nrScales_*nrDirs, nrX_*nrY_, nrScales_, speed);

	// half-wave rectification to avoid "negative firing rates"
	// 0 Hz spontaneous firing
	// TODO: justify scaling factors 只对正的值乘以倍数，负的值*1是否会加剧噪声
	dev_scaleHalfRect<<<iDivUp(nrX_*nrY_*nrDirs,128), 128>>>(d_respV1c, nrX_*nrY_*nrDirs, scaleV1ComplexFiring_, 1.0);

       // scale to firing rate
//       dev_scale<<<iDivUp(nrX_*nrY_*nrDirs,128), 128>>>(d_respV1c, scaleV1ComplexFiring_);
}


/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

// OMS: apply K2/K3/K4 Gaussian center-surround DoG to activitySurface,
//      OR their positive-threshold masks, and optionally pre-gate V1 input.
void MotionEnergy::applyOMS(bool gateInput) {
	const int len = nrX_ * nrY_;
	uint3 sizes = make_uint3(nrX_, nrY_, 1);

	CUDA_CHECK_ERRORS(cudaMemset(d_omsMask, 0, sizeof(unsigned char)*len));

	for (int scale = 0; scale < 3; scale++) {
		// 1. Gaussian center convolution on activitySurface -> d_omsCenter
		{
			CUDA_CHECK_ERRORS(cudaMemcpy(d_omsCenter, d_activitySurface,
				sizeof(float)*len, cudaMemcpyDeviceToDevice));
			conv2DGeneral(d_omsCenter, d_omsTmp_, sizes, d_multiCenterGauss_[scale], multiCenterSize_[scale]);
		}

		// 2. Gaussian surround convolution on activitySurface -> d_omsSurround
		{
			CUDA_CHECK_ERRORS(cudaMemcpy(d_omsSurround, d_activitySurface,
				sizeof(float)*len, cudaMemcpyDeviceToDevice));
			conv2DGeneral(d_omsSurround, d_omsTmp_, sizes, d_multiSurroundGauss_[scale], multiSurroundSize_[scale]);
		}

		// 3. DoG = center - surround, store in d_omsCenter
		{
			dev_sub<<<iDivUp(len, 256), 256>>>(d_omsCenter, d_omsSurround, d_omsCenter, len);
			CUDA_GET_LAST_ERROR("applyOMS/dev_sub (DoG) failed\n");
		}

		if (scale == 0) {
			CUDA_CHECK_ERRORS(cudaMemcpy(d_omsDogDebug, d_omsCenter,
				sizeof(float)*len, cudaMemcpyDeviceToDevice));
		} else {
			dev_maxInPlace<<<iDivUp(len, 256), 256>>>(d_omsDogDebug, d_omsCenter, len);
			CUDA_GET_LAST_ERROR("applyOMS/dev_maxInPlace failed\n");
		}

		// 4. Find global max of this scale's DoG
		float maxVal;
		{
			int blockSize = 256;
			int nBlocks = iDivUp(len, blockSize);
			dev_reduceMinMax<<<nBlocks, blockSize, 2 * blockSize * sizeof(float)>>>(
				d_omsCenter, d_minBlock_, d_maxBlock_, len, d_nBlocks_);
			CUDA_GET_LAST_ERROR("applyOMS/dev_reduceMinMax failed\n");

			int nBlocksHost;
			CUDA_CHECK_ERRORS(cudaMemcpy(&nBlocksHost, d_nBlocks_, sizeof(int), cudaMemcpyDeviceToHost));

			dev_finalMinMax<<<1, blockSize, 2 * blockSize * sizeof(float)>>>(
				d_minBlock_, d_maxBlock_, d_minBlock_, d_maxBlock_, nBlocksHost);
			CUDA_GET_LAST_ERROR("applyOMS/dev_finalMinMax failed\n");

			CUDA_CHECK_ERRORS(cudaMemcpy(&maxVal, d_maxBlock_, sizeof(float), cudaMemcpyDeviceToHost));
		}

		// 5. OR this scale's positive-threshold mask into d_omsMask.
		{
			dev_omsThresholdPositiveOr<<<iDivUp(len, 256), 256>>>(
				d_omsCenter, d_omsMaskScale, d_omsMask,
				maxVal, multiThreshold_[scale], len);
			CUDA_GET_LAST_ERROR("applyOMS/dev_omsThresholdPositiveOr failed\n");
		}
	}

	// 6. Legacy mode gates V1 input.  Mask-only mode deliberately preserves the
	// raw signed event surface so object suppression can happen after MT.
	if (gateInput) {
		dev_omsGateSoft<<<iDivUp(len, 256), 256>>>(
			d_eventSurface, d_omsMask,
			&d_stimBuf[nrX_ * nrY_ * (nrT - 1)],
			omsBackgroundKeep_, len);
		CUDA_GET_LAST_ERROR("applyOMS/dev_omsGateSoft failed\n");
	} else {
		CUDA_CHECK_ERRORS(cudaMemcpy(&d_stimBuf[nrX_*nrY_*(nrT-1)], d_eventSurface,
			sizeof(float)*len, cudaMemcpyDeviceToDevice));
	}
}

void MotionEnergy::buildGaussianKernel(float* kernel, int size, float sigma) {
	if (sigma < 1e-6f) {
		// delta at center
		for (int i = 0; i < size; i++) kernel[i] = 0.0f;
		kernel[size / 2] = 1.0f;
		return;
	}
	float center = (size - 1) * 0.5f;
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		float x = i - center;
		kernel[i] = expf(-0.5f * (x / sigma) * (x / sigma));
		sum += kernel[i];
	}
	if (sum > 0.0f) {
		for (int i = 0; i < size; i++) kernel[i] /= sum;
	}
}

void MotionEnergy::initME() {
//	std::stringstream minVal; minVal << min_nrX_ << " pixels.";
//	UserErrors::assertTrue(nrX_>=min_nrX_, UserErrors::CANNOT_BE_SMALLER, "initME()", "nrX", minVal.str());
//	minVal.str(""); minVal << min_nrY_ << " pixels.";
//	UserErrors::assertTrue(nrY_>=min_nrY_, UserErrors::CANNOT_BE_SMALLER, "initME()", "nrX", minVal.str());

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_resp_, sizeof(float)*nrX_*nrY_*nrFilters*nrScales_)); // V1 filter responses
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_respV1c, sizeof(float)*nrX_*nrY_*nrDirs)); // V1 complex cell responses

	CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_stim, nrX_*nrY_*nrC_));
	CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_stimBuf, nrX_*nrY_*nrT*sizeof(float)));
	CUDA_CHECK_ERRORS(cudaMemset (d_stimBuf, 0, nrX_*nrY_*nrT*sizeof(float)));

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&diffV1GausBufT, sizeof(float)*nrX_*nrY_*v1GaussFiltSize));

	CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_scalingStimBuf, nrX_*nrY_*nrT*sizeof(float)));
	CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_v1GausBuf, nrX_*nrY_*nrT*sizeof(float)));
	CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_diffV1GausBuf, nrX_*nrY_*nrT*sizeof(float)));
	CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_pop, nrX_*sizeof(float)*nrY_*nrScales_)); // mean of 28 filter responses for all x,y and spatial scales, at a given step in time

	// OMS GPU buffer allocation
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_eventSurface, sizeof(float)*nrX_*nrY_));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_activitySurface, sizeof(float)*nrX_*nrY_));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_omsCenter, sizeof(float)*nrX_*nrY_));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_omsSurround, sizeof(float)*nrX_*nrY_));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_omsDogDebug, sizeof(float)*nrX_*nrY_));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_omsMask, sizeof(unsigned char)*nrX_*nrY_));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_omsMaskScale, sizeof(unsigned char)*nrX_*nrY_));
	CUDA_CHECK_ERRORS(cudaMemset(d_eventSurface, 0, sizeof(float)*nrX_*nrY_));
	CUDA_CHECK_ERRORS(cudaMemset(d_activitySurface, 0, sizeof(float)*nrX_*nrY_));

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_onFrame, nrX_*nrY_));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_offFrame, nrX_*nrY_));

	// OMS Gaussian kernels (tuned for ~20px target in 108x108)
	// center covers the object, surround provides context
	centerGaussSize_ = 21;
	surroundGaussSize_ = 31;
	omsCenterSigma_ = 3.5f;
	omsSurroundSigma_ = 6.0f;

	std::vector<float> h_center(centerGaussSize_);
	std::vector<float> h_surround(surroundGaussSize_);
	buildGaussianKernel(h_center.data(), centerGaussSize_, omsCenterSigma_);
	buildGaussianKernel(h_surround.data(), surroundGaussSize_, omsSurroundSigma_);

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_centerGauss_, sizeof(float)*centerGaussSize_));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_surroundGauss_, sizeof(float)*surroundGaussSize_));
	CUDA_CHECK_ERRORS(cudaMemcpy(d_centerGauss_, h_center.data(), sizeof(float)*centerGaussSize_, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMemcpy(d_surroundGauss_, h_surround.data(), sizeof(float)*surroundGaussSize_, cudaMemcpyHostToDevice));

	multiCenterSize_[0] = 21; multiCenterSigma_[0] = 4.0f;
	multiSurroundSize_[0] = 41; multiSurroundSigma_[0] = 8.0f; multiThreshold_[0] = 0.60f;
	multiCenterSize_[1] = 31; multiCenterSigma_[1] = 6.0f;
	multiSurroundSize_[1] = 61; multiSurroundSigma_[1] = 12.0f; multiThreshold_[1] = 0.40f;
	multiCenterSize_[2] = 41; multiCenterSigma_[2] = 8.0f;
	multiSurroundSize_[2] = 81; multiSurroundSigma_[2] = 16.0f; multiThreshold_[2] = 0.60f;

	for (int s = 0; s < 3; s++) {
		std::vector<float> h_multiCenter(multiCenterSize_[s]);
		std::vector<float> h_multiSurround(multiSurroundSize_[s]);
		buildGaussianKernel(h_multiCenter.data(), multiCenterSize_[s], multiCenterSigma_[s]);
		buildGaussianKernel(h_multiSurround.data(), multiSurroundSize_[s], multiSurroundSigma_[s]);
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_multiCenterGauss_[s], sizeof(float)*multiCenterSize_[s]));
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_multiSurroundGauss_[s], sizeof(float)*multiSurroundSize_[s]));
		CUDA_CHECK_ERRORS(cudaMemcpy(d_multiCenterGauss_[s], h_multiCenter.data(), sizeof(float)*multiCenterSize_[s], cudaMemcpyHostToDevice));
		CUDA_CHECK_ERRORS(cudaMemcpy(d_multiSurroundGauss_[s], h_multiSurround.data(), sizeof(float)*multiSurroundSize_[s], cudaMemcpyHostToDevice));
	}

	// OMS conv2D workspace
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_omsTmp_, sizeof(float)*nrX_*nrY_));

	// OMS min/max reduction workspace (max blocks = 1024 to be safe)
	const int maxBlocks = 1024;
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_minBlock_, sizeof(float)*maxBlocks));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_maxBlock_, sizeof(float)*maxBlocks));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_nBlocks_, sizeof(int)));

	// OMS default parameters
	eventDecay_ = 0.85f;
	eventGain_ = 2.0f;
	omsThreshold_ = 0.0f;  // 0 = pass everything; tune upward (0.3-0.8) to suppress background
	omsBackgroundKeep_ = 0.1f;
	omsMode_ = OMS_PRE_GATE;
	omsDebug_ = false;
	omsDebugInterval_ = 1;
	omsDebugFrame_ = 0;
	omsDebugPrefix_ = "oms_debug";

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&scalingFilt, sizeof(h_scalingFilt)));
	CUDA_CHECK_ERRORS(cudaMemcpy(scalingFilt, h_scalingFilt, sizeof(h_scalingFilt), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&v1Gaus, sizeof(h_v1GaussFilt)));
	CUDA_CHECK_ERRORS(cudaMemcpy(v1Gaus, h_v1GaussFilt, sizeof(h_v1GaussFilt), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&complexV1Filt, sizeof(h_complexV1Filt)));
	CUDA_CHECK_ERRORS(cudaMemcpy(complexV1Filt, h_complexV1Filt, sizeof(h_complexV1Filt), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&normV1filt, sizeof(h_normV1filt)));
	CUDA_CHECK_ERRORS(cudaMemcpy(normV1filt, h_normV1filt, sizeof(h_normV1filt), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&diff1filt, sizeof(h_diff1filt)));
	CUDA_CHECK_ERRORS(cudaMemcpy(diff1filt, h_diff1filt, sizeof(h_diff1filt), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&diff2filt, sizeof(h_diff2filt)));
	CUDA_CHECK_ERRORS(cudaMemcpy(diff2filt, h_diff2filt, sizeof(h_diff2filt), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&diff3filt, sizeof(h_diff3filt)));
	CUDA_CHECK_ERRORS(cudaMemcpy(diff3filt, h_diff3filt, sizeof(h_diff3filt), cudaMemcpyHostToDevice));

}

// init all scaling factors, akin to shPars.m
void MotionEnergy::initParams() {
	scaleV1Linear_ = 6.6084;//input to simple V1 6.6084
	scaleV1FullWaveRect_ = 1.9263;//1.9263
	scaleV1Blur_ = 1.0205;//1.0205
	scaleV1NormPopK_ = 1; //0.2401;
	scaleV1NormStrength_ = 0.98;
	scaleV1Complex_ = 25;//0.99
	scaleV1C50_ = 0.1;
	scaleV1ComplexFiring_ = 10.0;//

}

void MotionEnergy::setOmsGaussianParams(int centerSize, float centerSigma,
                                        int surroundSize, float surroundSigma) {
	// Free old kernels if reallocating
	if (d_centerGauss_) CUDA_CHECK_ERRORS(cudaFree(d_centerGauss_));
	if (d_surroundGauss_) CUDA_CHECK_ERRORS(cudaFree(d_surroundGauss_));

	centerGaussSize_ = centerSize;
	surroundGaussSize_ = surroundSize;
	omsCenterSigma_ = centerSigma;
	omsSurroundSigma_ = surroundSigma;

	std::vector<float> h_center(centerGaussSize_);
	std::vector<float> h_surround(surroundGaussSize_);
	buildGaussianKernel(h_center.data(), centerGaussSize_, omsCenterSigma_);
	buildGaussianKernel(h_surround.data(), surroundGaussSize_, omsSurroundSigma_);

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_centerGauss_, sizeof(float)*centerGaussSize_));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_surroundGauss_, sizeof(float)*surroundGaussSize_));
	CUDA_CHECK_ERRORS(cudaMemcpy(d_centerGauss_, h_center.data(), sizeof(float)*centerGaussSize_, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMemcpy(d_surroundGauss_, h_surround.data(), sizeof(float)*surroundGaussSize_, cudaMemcpyHostToDevice));

	// Historically these parameters rebuilt only kernels that applyOMS did not
	// use.  Keep the API, but make it control all active OMS scales.
	for (int scale = 0; scale < 3; scale++) {
		setOmsMultiScaleParams(scale, centerSize, centerSigma,
			surroundSize, surroundSigma, multiThreshold_[scale]);
	}
}

void MotionEnergy::setOmsThreshold(float threshold) {
	assert(threshold >= 0.0f && threshold <= 1.0f);
	omsThreshold_ = threshold;
	for (int scale = 0; scale < 3; scale++) multiThreshold_[scale] = threshold;
}

void MotionEnergy::setOmsMultiScaleParams(int scaleIndex,
		int centerSize, float centerSigma,
		int surroundSize, float surroundSigma,
		float threshold) {
	assert(scaleIndex >= 0 && scaleIndex < 3);
	assert(centerSize > 0 && centerSize <= 121 && centerSize % 2 == 1);
	assert(surroundSize > 0 && surroundSize <= 121 && surroundSize % 2 == 1);
	assert(centerSigma > 0.0f && surroundSigma > 0.0f);
	assert(threshold >= 0.0f && threshold <= 1.0f);

	if (d_multiCenterGauss_[scaleIndex])
		CUDA_CHECK_ERRORS(cudaFree(d_multiCenterGauss_[scaleIndex]));
	if (d_multiSurroundGauss_[scaleIndex])
		CUDA_CHECK_ERRORS(cudaFree(d_multiSurroundGauss_[scaleIndex]));

	multiCenterSize_[scaleIndex] = centerSize;
	multiCenterSigma_[scaleIndex] = centerSigma;
	multiSurroundSize_[scaleIndex] = surroundSize;
	multiSurroundSigma_[scaleIndex] = surroundSigma;
	multiThreshold_[scaleIndex] = threshold;

	std::vector<float> center(centerSize);
	std::vector<float> surround(surroundSize);
	buildGaussianKernel(center.data(), centerSize, centerSigma);
	buildGaussianKernel(surround.data(), surroundSize, surroundSigma);
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_multiCenterGauss_[scaleIndex], sizeof(float)*centerSize));
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_multiSurroundGauss_[scaleIndex], sizeof(float)*surroundSize));
	CUDA_CHECK_ERRORS(cudaMemcpy(d_multiCenterGauss_[scaleIndex], center.data(),
		sizeof(float)*centerSize, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMemcpy(d_multiSurroundGauss_[scaleIndex], surround.data(),
		sizeof(float)*surroundSize, cudaMemcpyHostToDevice));
}

void MotionEnergy::copyOmsMask(unsigned char* hostMask) const {
	assert(hostMask != NULL);
	CUDA_CHECK_ERRORS(cudaMemcpy(hostMask, d_omsMask,
		sizeof(unsigned char)*nrX_*nrY_, cudaMemcpyDeviceToHost));
}

void MotionEnergy::setOmsDebug(bool enabled, int interval, const char* prefix) {
	omsDebug_ = enabled;
	omsDebugInterval_ = interval > 0 ? interval : 1;
	if (prefix && prefix[0] != '\0') {
		omsDebugPrefix_ = prefix;
	}
}

void MotionEnergy::dumpFloatPgm(const char* path, const float* d_data) {
	const int len = nrX_ * nrY_;
	std::vector<float> h_data(len);
	CUDA_CHECK_ERRORS(cudaMemcpy(h_data.data(), d_data, sizeof(float) * len, cudaMemcpyDeviceToHost));

	float minVal = FLT_MAX;
	float maxVal = -FLT_MAX;
	for (int i = 0; i < len; i++) {
		float v = h_data[i];
		if (v < minVal) minVal = v;
		if (v > maxVal) maxVal = v;
	}

	float range = maxVal - minVal;
	if (range < 1e-12f) range = 1.0f;

	FILE* f = fopen(path, "wb");
	if (!f) {
		fprintf(stderr, "[OMS debug] WARNING: cannot write %s\n", path);
		return;
	}
	fprintf(f, "P5\n%d %d\n255\n", nrX_, nrY_);
	for (int i = 0; i < len; i++) {
		float norm = (h_data[i] - minVal) / range;
		if (norm < 0.0f) norm = 0.0f;
		if (norm > 1.0f) norm = 1.0f;
		unsigned char px = static_cast<unsigned char>(norm * 255.0f + 0.5f);
		fwrite(&px, 1, 1, f);
	}
	fclose(f);
}

void MotionEnergy::dumpMaskPgm(const char* path, const unsigned char* d_data) {
	const int len = nrX_ * nrY_;
	std::vector<unsigned char> h_data(len);
	CUDA_CHECK_ERRORS(cudaMemcpy(h_data.data(), d_data, len, cudaMemcpyDeviceToHost));

	FILE* f = fopen(path, "wb");
	if (!f) {
		fprintf(stderr, "[OMS debug] WARNING: cannot write %s\n", path);
		return;
	}
	fprintf(f, "P5\n%d %d\n255\n", nrX_, nrY_);
	fwrite(h_data.data(), 1, len, f);
	fclose(f);
}

void MotionEnergy::dumpOmsDebugFrame() {
	char path[512];
	snprintf(path, sizeof(path), "%s_%06d_activity.pgm", omsDebugPrefix_.c_str(), omsDebugFrame_);
	dumpFloatPgm(path, d_activitySurface);
	snprintf(path, sizeof(path), "%s_%06d_dog.pgm", omsDebugPrefix_.c_str(), omsDebugFrame_);
	dumpFloatPgm(path, d_omsDogDebug);
	snprintf(path, sizeof(path), "%s_%06d_mask.pgm", omsDebugPrefix_.c_str(), omsDebugFrame_);
	dumpMaskPgm(path, d_omsMask);
	snprintf(path, sizeof(path), "%s_%06d_gated.pgm", omsDebugPrefix_.c_str(), omsDebugFrame_);
	dumpFloatPgm(path, &d_stimBuf[nrX_ * nrY_ * (nrT - 1)]);
}

// get the responses of the filters specified in d_v1popDirs by interpolation
// this is basically doing what shSwts.m did in the original S&H matlab code
void MotionEnergy::accumDiffStims(float *d_resp_tmp, float* diffV1GausBuf, dim3 _sizes, int orderX, int orderY, int orderT) {
	// a useful list of factorials for computing the scaling factors for the derivatives
	int factorials[4] = {1, 1, 2, 6};

	// the scaling factor for this directial derivative; similar to the binomial coefficients
	int scale = 6/factorials[orderX]/factorials[orderY]/factorials[orderT];

	dev_accumDiffStims<<<iDivUp(_sizes.x*_sizes.y, 256), 256>>>(d_resp_tmp, diffV1GausBuf, _sizes.x*_sizes.y, scale,
		orderX, orderY, orderT);
	CUDA_GET_LAST_ERROR("dev_accumDiffStims() execution failed\n");
}

// odata must be pre-allocated
// the result will end up in idata...
// filtlen can not be greater than CONVN_THREAD_SIZE2
void MotionEnergy::conv2D(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen) {
	assert(filtlen <= CONVN_THREAD_SIZE2);
	unsigned int* sizes = (unsigned int*)&_sizes;
	float* tmp;

	// convolve the first dimension
	dim3 grid1(iDivUp(sizes[0], CONV1_THREAD_SIZE-(filtlen-1)), sizes[1]*sizes[2]);
	dim3 threads1(CONV1_THREAD_SIZE, 1, 1);
	dev_conv1<<<grid1, threads1>>>(idata, odata, sizes[0], filt, filtlen);
        CUDA_GET_LAST_ERROR("dev_conv1() execution failed\n");

	tmp = idata;
	idata = odata;
	odata = tmp;

	// convolve the second dimension
	dim3 grid2(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[1], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[2]);
	dim3 threads2(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
	dev_convn<<<grid2, threads2>>>(idata, odata, sizes[0], sizes[1], sizes[0], sizes[0]*sizes[1], sizes[2], filt, filtlen);
        CUDA_GET_LAST_ERROR("dev_convn() execution failed\n");
}

void MotionEnergy::conv2DGeneral(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen) {
	unsigned int* sizes = (unsigned int*)&_sizes;
	int total = sizes[0] * sizes[1] * sizes[2];

	dev_convRowsGeneral<<<iDivUp(total, 256), 256>>>(idata, odata, sizes[0], sizes[1], sizes[2], filt, filtlen);
	CUDA_GET_LAST_ERROR("dev_convRowsGeneral() execution failed\n");

	dev_convColsGeneral<<<iDivUp(total, 256), 256>>>(odata, idata, sizes[0], sizes[1], sizes[2], filt, filtlen);
	CUDA_GET_LAST_ERROR("dev_convColsGeneral() execution failed\n");
}

// conv3D is only used in the motion model in freq space (\omega_x,\omega_y,\omega_t)
// odata must be pre-allocated
// the result will end up in idata
// filtlen can not be greater than CONVN_THREAD_SIZE2
void MotionEnergy::conv3D(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen) {
	assert(filtlen <= CONVN_THREAD_SIZE2);
	unsigned int* sizes = (unsigned int*)&_sizes;
	float* tmp;

	// convolve the first dimension
	dim3 grid1(iDivUp(sizes[0], CONV1_THREAD_SIZE-(filtlen-1)), sizes[1]*sizes[2]);
	dim3 threads1(CONV1_THREAD_SIZE, 1, 1);
	dev_conv1<<<grid1, threads1>>>(idata, odata, sizes[0], filt, filtlen);
		CUDA_GET_LAST_ERROR("dev_conv1() execution failed\n");
	tmp = idata;
	idata = odata;
	odata = tmp;

	// convolve the second dimension
	dim3 grid2(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[1], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[2]);
	dim3 threads2(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
	dev_convn<<<grid2, threads2>>>(idata, odata, sizes[0], sizes[1], sizes[0], sizes[0]*sizes[1], sizes[2], filt, filtlen);
		CUDA_GET_LAST_ERROR("dev_convn() execution failed\n");

	tmp = idata;
	idata = odata;
	odata = tmp;

	// convolve the third dimension
	dim3 grid3(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[2], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[1]);
	dim3 threads3(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
	dev_convn<<<grid3, threads3>>>(idata, odata, sizes[0], sizes[2], sizes[0]*sizes[1], sizes[0], sizes[1], filt, filtlen);
		CUDA_GET_LAST_ERROR("dev_convn() execution failed\n");

	tmp = idata;
	idata = odata;
	odata = tmp;
}

//will free idata
// this computes the difference / approximates the derivative of idata
float* MotionEnergy::diff(float* idata, dim3 _sizes, int order, int dim)
{
	unsigned int* sizes = (unsigned int*)&_sizes;
	int filtlen;
	float* filt;
	float* odata;

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&odata, sizeof(float)*sizes[0]*sizes[1]*sizes[2]));

	switch (order) {
		case 1:
			filtlen = diff1filtSize;
			filt = diff1filt;
			break;
		case 2:
			filtlen = diff2filtSize;
			filt = diff2filt;
			break;
		case 3:
			filtlen = diff3filtSize;
			filt = diff3filt;
			break;
	}

	switch (dim) {
		case 0: {
			// convolve the first dimension
			dim3 grid1(iDivUp(sizes[0], CONV1_THREAD_SIZE-(filtlen-1)), sizes[1]*sizes[2]);
			dim3 threads1(CONV1_THREAD_SIZE, 1, 1);
			dev_conv1<<<grid1, threads1>>>(idata, odata, sizes[0], filt, filtlen);
			CUDA_GET_LAST_ERROR("dev_conv1() execution failed\n");
			break;
		}
		case 1: {
			// convolve the second dimension
			dim3 grid2(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[1], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[2]);
			dim3 threads2(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
			dev_convn<<<grid2, threads2>>>(idata, odata, sizes[0], sizes[1], sizes[0], sizes[0]*sizes[1], sizes[2], filt, filtlen);
			CUDA_GET_LAST_ERROR("dev_convn() execution failed\n");
			break;
		}
		case 2: {
			// convolve the third dimension
			dim3 grid3(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[2], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[1]);
			dim3 threads3(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
			dev_convn<<<grid3, threads3>>>(idata, odata, sizes[0], sizes[2], sizes[0]*sizes[1], sizes[0], sizes[1], filt, filtlen);
			CUDA_GET_LAST_ERROR("dev_convn() execution failed\n");
			break;
		}
	}

	CUDA_CHECK_ERRORS(cudaFree (idata));

	return odata;
}
