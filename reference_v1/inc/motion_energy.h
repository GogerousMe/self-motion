#ifndef _MOTION_ENERGY_H_
#define _MOTION_ENERGY_H_

#include <vector_types.h>			// dim3
#include <string>


class MotionEnergy {
public:
	enum OmsMode {
		OMS_OFF = 0,
		OMS_PRE_GATE = 1,
		OMS_MASK_ONLY = 2
	};

	MotionEnergy(int nrX, int nrY, int nrC);
	~MotionEnergy();

	void setNumChannels(int nrC);

	void calcV1complex(unsigned char* stim, float* V1comp, double speed=1.0, bool GPUpointers=true);

	// event-based entry point (replaces calcV1complex for event camera data)
	void calcV1complexEvents(
		unsigned char* onEvents,
		unsigned char* offEvents,
		float* V1comp,
		double speed=1.0,
		bool GPUpointers=true
	);

	double getScaleV1Linear() { return scaleV1Linear_; }
	double getScaleV1FullWaveRect() { return scaleV1FullWaveRect_; }
	double getScaleV1Blur() { return scaleV1Blur_; }
	double getScaleV1NormPopK() { return scaleV1NormPopK_; }
	double getScaleV1NormStrength() { return scaleV1NormStrength_; }
	double getScaleV1Complex() { return scaleV1Complex_; }
	double getScaleV1ComplexFiring() { return scaleV1ComplexFiring_; }
	double getScaleV1C50() { return scaleV1C50_; }

	void setScaleV1Linear(double s) { scaleV1Linear_ = s; }
	void setScaleV1FullWaveRect(double s) { scaleV1FullWaveRect_ = s; }
	void setScaleV1Blur(double s) { scaleV1Blur_ = s; }
	void setScaleV1NormPopK(double s) { scaleV1NormPopK_ = s; }
	void setScaleV1NormStrength(double s) { scaleV1NormStrength_ = s; }
	void setScaleV1Complex(double s) { scaleV1Complex_ = s; }
	void setScaleV1ComplexFiring(double s) { scaleV1ComplexFiring_ = s; }
	void setScaleV1C50(double s) { scaleV1C50_ = s; }

	// OMS parameter getters/setters
	float getEventDecay()       const { return eventDecay_; }
	float getEventGain()        const { return eventGain_; }
	float getOmsThreshold()     const { return omsThreshold_; }
	float getOmsBackgroundKeep() const { return omsBackgroundKeep_; }
	int   getOmsCenterSize()    const { return centerGaussSize_; }
	int   getOmsSurroundSize()  const { return surroundGaussSize_; }
	float getOmsCenterSigma()   const { return omsCenterSigma_; }
	float getOmsSurroundSigma() const { return omsSurroundSigma_; }
	OmsMode getOmsMode() const { return omsMode_; }

	void setEventDecay(float v)       { eventDecay_ = v; }
	void setEventGain(float v)        { eventGain_ = v; }
	void setOmsThreshold(float v);
	void setOmsBackgroundKeep(float v) { omsBackgroundKeep_ = v; }
	void setOmsMode(OmsMode mode) { omsMode_ = mode; }
	void setOmsDebug(bool enabled, int interval=1, const char* prefix="oms_debug");
	void copyOmsMask(unsigned char* hostMask) const;

	// Rebuild Gaussian kernels (call before calcV1complexEvents)
	void setOmsGaussianParams(int centerSize, float centerSigma,
	                          int surroundSize, float surroundSigma);
	void setOmsMultiScaleParams(int scaleIndex,
	                            int centerSize, float centerSigma,
	                            int surroundSize, float surroundSigma,
	                            float threshold);

private:
	void initME();
	void initParams(); // shPars.m
	void accumDiffStims(float *d_resp_tmp, float* diffV1GausBuf, dim3 _sizes, int orderX, int orderY, int orderT);
	void conv2D(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen);
	void conv2DGeneral(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen);
	void conv3D(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen);
	float* diff(float* idata, dim3 _sizes, int order, int dim);

	void loadInput(unsigned char* stim);
	void calcV1linear();
	void calcV1rect();
	void calcV1blur();
	void calcV1normalize();
	void calcV1direction(double speed);
	void applyOMS(bool gateInput);
	void buildGaussianKernel(float* kernel, int size, float sigma);
	void dumpOmsDebugFrame();
	void dumpFloatPgm(const char* path, const float* d_data);
	void dumpMaskPgm(const char* path, const unsigned char* d_data);

	int nrX_;
	int nrY_;
	int nrC_;

	int nrScales_;

	int min_nrX_;
	int min_nrY_;

	// from shPars.m
	double scaleV1Linear_;
	double scaleV1FullWaveRect_;
	double scaleV1Blur_;
	double scaleV1NormPopK_;
	double scaleV1NormStrength_;
	double scaleV1Complex_;
	double scaleV1C50_;
	double scaleV1ComplexFiring_;

	int stimChannels;

	float* scalingFilt;
	float* v1Gaus;
	float* complexV1Filt;
	float* normV1filt;
	float* diff1filt;
	float* diff2filt;
	float* diff3filt;

	float* d_resp_; // the returned V1ME filter responses
	float* d_respV1c; // the returned V1 complex cell responses
	float* d_stimBuf;
	float* d_scalingStimBuf; // the temporary matrix that will be filtered once for each scale...
	float* d_v1GausBuf;
	float* d_v1GausBuf2;
	float* d_diffV1GausBuf;
	float* diffV1GausBufT;

	unsigned char* d_stim; // the video input
	float* d_pop;

	// OMS GPU buffers
	float* d_eventSurface;
	float* d_activitySurface;
	float* d_omsCenter;    // center conv result / DoG workspace
	float* d_omsSurround;  // surround conv result
	float* d_omsDogDebug;  // max DoG over multi-scale OMS for debug output
	unsigned char* d_omsMask;  // binary mask (0 or 255)
	unsigned char* d_omsMaskScale;  // per-scale temporary mask

	// Gaussian kernels for OMS (1D profiles, for separable conv2D)
	float* d_centerGauss_;
	float* d_surroundGauss_;
	int centerGaussSize_;
	int surroundGaussSize_;
	float omsCenterSigma_;
	float omsSurroundSigma_;
	float* d_multiCenterGauss_[3];
	float* d_multiSurroundGauss_[3];
	int multiCenterSize_[3];
	int multiSurroundSize_[3];
	float multiCenterSigma_[3];
	float multiSurroundSigma_[3];
	float multiThreshold_[3];

	// conv2D temp buffer + min/max reduction workspace
	float* d_omsTmp_;
	float* d_minBlock_;
	float* d_maxBlock_;
	int*   d_nBlocks_;

	// event input GPU buffers
	unsigned char* d_onFrame;
	unsigned char* d_offFrame;

	// OMS parameters
	float eventDecay_;
	float eventGain_;
	float omsThreshold_;
	float omsBackgroundKeep_;
	OmsMode omsMode_;
	bool omsDebug_;
	int omsDebugInterval_;
	int omsDebugFrame_;
	std::string omsDebugPrefix_;
};

#endif
