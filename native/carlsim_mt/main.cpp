// Offline bridge to the real CARLsim engine and historical three-group layout.
// Run in a NEW working directory; all rates/weights paths must be absolute.
#include <carlsim.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <cstdlib>
#endif

static int integer(const char* s) {
    size_t end=0; long v=std::stol(s,&end);
    if (s[end] || v<0 || v>std::numeric_limits<int>::max()) throw std::runtime_error("invalid integer");
    return static_cast<int>(v);
}
static double elapsed(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
}
class SignedProjection: public ConnectionGenerator {
    float (*weights)[8]; int pixels; bool positive,mapped;
public:
    SignedProjection(float (*a)[8],int p,bool pos,bool legacy):weights(a),pixels(p),positive(pos),mapped(legacy) {}
    void connect(CARLsim*,int,int i,int,int j,float& weight,float& maxWt,float& delay,bool& connected) {
        int f=mapped?i:(i/pixels)%28;
        int d=mapped?j:j/pixels;
        if(f<0 || f>=28 || d<0 || d>=8) throw std::runtime_error("callback layout mismatch");
        bool local=mapped?connected:(i%pixels==j%pixels);
        float a=weights[f][d];
        connected=local && (positive?a>0:a<0);
        weight=maxWt=1.5f*std::fabs(a);delay=1;
    }
};
int main(int argc,char** argv) { try {
#ifdef _WIN32
    _set_fmode(_O_BINARY); // CARLsim opens DAT with "w+"; prevent CRLF expansion.
#endif
    if(argc!=12) throw std::runtime_error("usage: carlsim_mt rates.f32 weights.f32 W H frames frame_ms seed cpu|gpu v1_scale inhibition_0_or_1 legacy_mapped|standard");
    const int w=integer(argv[3]),h=integer(argv[4]),frames=integer(argv[5]),dt=integer(argv[6]),seed=integer(argv[7]);
    if(!w || !h || !frames || !dt || static_cast<int64_t>(frames)*dt>std::numeric_limits<int>::max()
       || static_cast<int64_t>(w)*h>std::numeric_limits<int>::max()/100)
        throw std::runtime_error("invalid dimensions or duration");
    const int p=w*h;
    const std::string layout=argv[11];
    if(layout!="legacy_mapped" && layout!="standard") throw std::runtime_error("unknown callback layout");
    const int inhibition=integer(argv[10]);
    if(inhibition>1) throw std::runtime_error("inhibition must be 0 or 1");
    const std::string mode=argv[8];
    if(mode!="cpu" && mode!="gpu") throw std::runtime_error("mode must be cpu or gpu");
    size_t end=0; const std::string scaleText=argv[9]; const float scale=std::stof(scaleText,&end);
    if(end!=scaleText.size() || !std::isfinite(scale) || scale<=0) throw std::runtime_error("invalid V1 scale");
    const uint32_t endian=1;
    if(*reinterpret_cast<const unsigned char*>(&endian)!=1 || sizeof(float)!=4)
        throw std::runtime_error("little-endian float32 host required");
    std::ifstream input(argv[1],std::ios::binary), weightsFile(argv[2],std::ios::binary);
    float weights[28][8];
    if(!weightsFile.read(reinterpret_cast<char*>(weights),sizeof(weights)) || weightsFile.peek()!=EOF)
        throw std::runtime_error("weights must be exactly 28x8 float32");
    for(int f=0;f<28;f++) for(int d=0;d<8;d++)
        if(!std::isfinite(weights[f][d])) throw std::runtime_error("nonfinite weight");
    input.seekg(0,std::ios::end);
    const int64_t bytes=static_cast<int64_t>(frames)*84*p*4;
    if(!input || input.tellg()!=bytes) throw std::runtime_error("rates file size mismatch");
    input.seekg(0);
    // Validate before allocating a network. No clipping/normalizing V1 rates.
    std::vector<float> plane(p);
    for(int64_t k=0;k<static_cast<int64_t>(frames)*84;k++) {
        input.read(reinterpret_cast<char*>(plane.data()),p*4);
        for(float r:plane) if(!std::isfinite(r) || r<0 || r*scale>1000.f)
            throw std::runtime_error("V1 rate outside finite [0,1000]Hz after scaling");
    }
    input.clear(); input.seekg(0);
    auto start=std::chrono::steady_clock::now();
    std::cerr<<"[setup] three-group MT; V1-to-MT candidate pairs="
             <<(static_cast<uint64_t>(84)*p*8*p*2)<<std::endl;
    uint64_t edges=0;
    std::vector<short> connectionIds;
    double setupSeconds=0,runSeconds=0;
    {
        CARLsim sim("mstd_mt",mode=="gpu"?GPU_MODE:CPU_MODE,SILENT,0,seed);
        int v1=sim.createSpikeGeneratorGroup("V1",Grid3D(w,h,84),EXCITATORY_NEURON);
        int exc=sim.createGroup("MT1CDS",Grid3D(w,h,8),EXCITATORY_NEURON);
        int inh=sim.createGroup("MT1CDSinh",Grid3D(w,h,8),INHIBITORY_NEURON);
        sim.setNeuronParameters(exc,0.02f,0.2f,-65.f,8.f);
        sim.setNeuronParameters(inh,0.02f,0.2f,-65.f,8.f);
        SignedProjection positive(weights,p,true,layout=="legacy_mapped");
        SignedProjection negative(weights,p,false,layout=="legacy_mapped");
        connectionIds.push_back(sim.connect(v1,exc,&positive,SYN_FIXED));
        connectionIds.push_back(sim.connect(v1,inh,&negative,SYN_FIXED));
        for(int f=0;f<28;f++) for(int d=0;d<8;d++) if(weights[f][d]!=0) edges+=3*p;
        if(inhibition) {
            connectionIds.push_back(sim.connect(inh,exc,"one-to-one",RangeWeight(1.f),1.f,RangeDelay(1)));
            edges+=8*p;
        }
        sim.setConductances(true);
        sim.setupNetwork();
        uint64_t actualEdges=0;
        for(short id:connectionIds) actualEdges+=sim.getNumSynapticConnections(id);
        if(actualEdges!=edges) throw std::runtime_error("actual synapse count differs from local signed topology");
        PoissonRate rates(84*p,false);
        sim.setSpikeMonitor(v1,"v1.dat");
        sim.setSpikeMonitor(exc,"exc.dat");
        sim.setSpikeMonitor(inh,"inh.dat");
        std::vector<float> frame(84*p);
        setupSeconds=elapsed(start);
        std::cerr<<"[setup done] seconds="<<setupSeconds<<", synapses="<<edges<<std::endl;
        start=std::chrono::steady_clock::now();
        for(int t=0;t<frames;t++) {
            if(!input.read(reinterpret_cast<char*>(frame.data()),84*p*4)) throw std::runtime_error("truncated rates");
            for(float& r:frame) r*=scale;
            rates.setRates(frame);
            sim.setSpikeRate(v1,&rates);
            sim.runNetwork(dt/1000,dt%1000,false);
            if(t==0 || (t+1)%10==0 || t+1==frames) std::cerr<<"[run] "<<t+1<<"/"<<frames<<std::endl;
        }
        runSeconds=elapsed(start);
    } // Flush all spike files before publishing completion.
    std::ofstream report("native_report.json");
    report.precision(17);
    report << "{\"schema\":\"mstd-carlsim-native-v1\",\"complete\":true,\"width\":"<<w
           <<",\"height\":"<<h<<",\"frames\":"<<frames<<",\"frame_ms\":"<<dt
           <<",\"seed\":"<<seed<<",\"mode\":\""<<mode<<"\",\"connections\":"<<edges
           <<",\"setup_seconds\":"<<setupSeconds<<",\"run_seconds\":"<<runSeconds<<"}";
    if(!report) throw std::runtime_error("report write failed");
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; return 1; } }
