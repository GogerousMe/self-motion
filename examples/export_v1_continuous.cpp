// Call the actual MotionEnergy library without constructing a CARLsim network.
// Linux x86_64, C++11. Output is raw SFYX float32 before spike sampling.
#include <motion_energy.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// Avoid std::filesystem: older CARLsim host toolchains lack that library.
// mkdir is atomic and refuses an existing directory, preserving capture files.
void create_output_directory(const std::string& path) {
#ifdef _WIN32
    const int result=_mkdir(path.c_str());
#else
    const int result=mkdir(path.c_str(),0777);
#endif
    if (result!=0)
        throw std::runtime_error("cannot create output (must not exist; parent must exist): "
                                 +std::string(std::strerror(errno)));
}

template<class T> T read_value(std::istream& input) {
    T value;
    if (!input.read(reinterpret_cast<char*>(&value),sizeof(value)))
        throw std::runtime_error("truncated stimulus header");
    return value;
}

int main(int argc,char** argv) {
    try {
        if (argc<4 || argc>5) {
            std::cerr<<"Usage: export_v1_continuous STIMULUS.dat NEW_OUTPUT_DIR FRAME_MS [MAX_FRAMES]\n";
            return 1;
        }
        const std::uint16_t endian=1;
        if (*reinterpret_cast<const unsigned char*>(&endian)!=1 || sizeof(float)!=4)
            throw std::runtime_error("little-endian float32 platform required");
        const double frame_ms=std::stod(argv[3]);
        if (!std::isfinite(frame_ms) || frame_ms<=0) throw std::runtime_error("invalid frame-ms");
        std::ifstream input(argv[1],std::ios::binary);
        if (!input) throw std::runtime_error("cannot open stimulus");
        const auto signature=read_value<std::int32_t>(input);
        const auto version=read_value<float>(input);
        const auto kind=read_value<std::int32_t>(input);
        const auto channels=read_value<unsigned char>(input);
        const auto width=read_value<std::int32_t>(input);
        const auto height=read_value<std::int32_t>(input);
        const auto total=read_value<std::int32_t>(input);
        // CARLsim categories: grating=0 ... picture=4, movie=5, compound=6.
        // All categories use uint8 image frames; kind is not a dtype selector.
        if (signature!=293390619 || version!=1.0f || kind<0 || kind>6 || width<19 || height<19 || total<=0
            || (channels!=1 && channels!=3)) throw std::runtime_error("unsupported uint8 VisualStimulus DAT");
        const std::size_t pixels=std::size_t(width)*height;
        const auto expected=25+std::uintmax_t(pixels)*channels*total;
        input.seekg(0,std::ios::end);
        const auto actual=input.tellg();
        if (actual<0 || static_cast<std::uintmax_t>(actual)!=expected)
            throw std::runtime_error("stimulus size mismatch");
        input.seekg(25,std::ios::beg);
        if (!input) throw std::runtime_error("cannot seek to stimulus frames");
        const int requested=argc==5?std::stoi(argv[4]):total;
        if (requested<=0) throw std::runtime_error("positive max-frames required");
        const int frames=std::min(total,requested);
        const std::string output(argv[2]);
        create_output_directory(output);
        std::ofstream rates(output+"/rates.f32",std::ios::binary);
        if (!rates) throw std::runtime_error("cannot create output");
        MotionEnergy me(width,height,channels);
        // Refuse an obvious wrong-gain library. Exact source/library hashes are bound by register-cpp.
        if (std::abs(me.getScaleV1Complex()-25.0)>1e-9)
            throw std::runtime_error("library complex gain differs from expected source profile");
        std::vector<unsigned char> frame(pixels*channels);
        std::vector<float> response(pixels*84,std::numeric_limits<float>::quiet_NaN());
        for (int i=0;i<frames;++i) {
            if (!input.read(reinterpret_cast<char*>(frame.data()),frame.size()))
                throw std::runtime_error("truncated stimulus");
            std::fill(response.begin(),response.end(),std::numeric_limits<float>::quiet_NaN());
            // false requests a synchronous device-to-host copy; speed projection is disabled in this profile.
            me.calcV1complex(frame.data(),response.data(),1.5,false);
            for (float value:response)
                if (!std::isfinite(value) || value<0) throw std::runtime_error("invalid/incomplete 84-channel output");
            rates.write(reinterpret_cast<const char*>(response.data()),response.size()*sizeof(float));
            if (!rates) throw std::runtime_error("response write failed");
            if (i==0 || (i+1)%20==0 || i+1==frames)
                std::cout<<"[C++ V1] "<<i+1<<"/"<<frames<<std::endl;
        }
        rates.close();
        if (!rates) throw std::runtime_error("response close failed");
        std::ofstream meta(output+"/capture.json");
        meta.precision(17);
        meta<<"{\"width\":"<<width<<",\"height\":"<<height<<",\"channels\":"<<int(channels)
            <<",\"frames\":"<<frames<<",\"frame_interval_s\":"<<frame_ms/1000.0<<"}\n";
        meta.close();
        if (!meta) throw std::runtime_error("capture metadata write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<std::endl;
        return 1;
    }
}
