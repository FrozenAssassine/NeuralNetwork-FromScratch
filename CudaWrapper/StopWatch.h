#pragma once
#include <chrono>

class Stopwatch {
public:
    Stopwatch(bool startNow = true) {
        if (startNow) Start();
    }

    void Start() {
        m_start = std::chrono::high_resolution_clock::now();
        m_running = true;
    }

    void Stop() {
        if (m_running) {
            m_end = std::chrono::high_resolution_clock::now();
            m_running = false;
        }
    }

    void Reset() {
        m_running = false;
        m_start = m_end = std::chrono::high_resolution_clock::now();
    }

    double ElapsedSeconds() const {
        return Elapsed<std::chrono::duration<double>>();
    }

    double ElapsedMilliseconds() const {
        return Elapsed<std::chrono::duration<double, std::milli>>();
    }

    double ElapsedMicroseconds() const {
        return Elapsed<std::chrono::duration<double, std::micro>>();
    }

private:
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_end;
    bool m_running = false;

    template <typename T>
    double Elapsed() const {
        auto endTime = m_running
            ? std::chrono::high_resolution_clock::now()
            : m_end;
        return std::chrono::duration_cast<T>(endTime - m_start).count();
    }
};
