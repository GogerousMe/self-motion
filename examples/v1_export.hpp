#pragma once
// Standalone exporter: host memory only, already direction-coded rates in [d,y,x].
// This does NOT call CARLsim or read GPU pointers. See docs/V1_INTERFACE.md.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <stdexcept>

inline void append_v1_direction_rates(std::ostream& out, const float* host_rates,
                                      std::size_t count, std::size_t expected_count) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "IEEE754 float32 required");
    if (!host_rates || count != expected_count)
        throw std::invalid_argument("V1 channel count mismatch");
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(host_rates[i]) || host_rates[i] < 0)
            throw std::invalid_argument("invalid V1 rate");
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, host_rates + i, sizeof(bits));
        const char bytes[4] = {static_cast<char>(bits & 255),
            static_cast<char>((bits >> 8) & 255), static_cast<char>((bits >> 16) & 255),
            static_cast<char>((bits >> 24) & 255)};
        out.write(bytes, 4);
    }
    if (!out) throw std::runtime_error("failed to write rates");
}
