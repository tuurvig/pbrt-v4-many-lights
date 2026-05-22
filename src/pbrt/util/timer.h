// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_TIMER_H
#define PBRT_UTIL_TIMER_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>


namespace pbrt {

// Timer Definition
class Timer {
  public:
    Timer() { start = clock::now(); }
    double ElapsedSeconds() const {
        clock::time_point now = clock::now();
        int64_t elapseduS =
            std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        return elapseduS / 1000000.;
    }

    int64_t ElapsedMicroseconds() const {
        clock::time_point now = clock::now();
        int64_t elapseduS = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        return elapseduS;
    }

    std::string ToString() const;

  private:
    using clock = std::chrono::steady_clock;
    clock::time_point start;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_TIMER_H
