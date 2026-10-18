// Protocol fixture only: artificial values and artificial intrinsics, no CARLsim.
// Usage: export_fixture <existing-empty-output-directory>
#include "v1_export.hpp"
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string prefix = std::string(argv[1]) + "/";
    std::ofstream data(prefix + "rates.f32", std::ios::binary);
    std::vector<float> frame(16);
    for (int t = 0; t < 2; ++t) {
        for (int i = 0; i < 16; ++i) frame[i] = static_cast<float>(t * 16 + i);
        append_v1_direction_rates(data, frame.data(), frame.size(), 16);
    }
    std::ofstream metadata(prefix + "metadata.json");
    metadata << R"({
      "schema":"v1-rate-v1", "encoding":"direction_rates",
      "dtype":"float32-le", "layout":"TDYX", "units":"Hz",
      "coordinates":"x-right_y-down", "timestamp_reference":"frame_start",
      "sample_hold":"until_next_timestamp", "width":2, "height":2,
      "directions_rad":[0.0,1.5707963267948966,3.141592653589793,4.71238898038469],
      "intrinsics_px":{"fx_px":2.0,"fy_px":2.0,"cx_px":0.5,"cy_px":0.5},
      "timestamps_s":[0.0,0.05], "provenance":{"source":"cpp_protocol_fixture"}
    })";
    return data && metadata ? 0 : 1;
}
