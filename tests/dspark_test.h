// DSPark Test Framework - Minimal, zero-dependency test harness
// Auto-registration, DSP-specific assertions, pass/fail counting.

#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

namespace dspark::test {

// ============================================================================
// Test registry
// ============================================================================

struct TestCase
{
    std::string name;
    std::function<void()> func;
};

inline std::vector<TestCase>& registry()
{
    static std::vector<TestCase> r;
    return r;
}

inline int& passCount() { static int n = 0; return n; }
inline int& failCount() { static int n = 0; return n; }
inline bool& currentTestFailed() { static bool f = false; return f; }

struct AutoRegister
{
    AutoRegister(const char* name, std::function<void()> func)
    {
        registry().push_back({ name, std::move(func) });
    }
};

// ============================================================================
// Runner
// ============================================================================

inline int runAll()
{
    auto& tests = registry();
    int passed = 0;
    int failed = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (auto& tc : tests)
    {
        // Named and flushed before the case runs, so a hard crash (signal,
        // stack exhaustion, an assertion firing) still identifies the culprit.
        // CTest buffers this and only prints it when the test fails, so a
        // green run stays silent.
        std::cout << "[ " << (passed + failed + 1) << "/" << tests.size() << " ] "
                  << tc.name << std::endl;

        currentTestFailed() = false;
        try
        {
            tc.func();
        }
        catch (const std::exception& e)
        {
            std::cerr << "  EXCEPTION [" << tc.name << "] " << e.what() << "\n";
            currentTestFailed() = true;
        }
        catch (...)
        {
            std::cerr << "  EXCEPTION [" << tc.name << "] unknown\n";
            currentTestFailed() = true;
        }

        if (currentTestFailed())
        {
            std::cerr << "  FAIL: " << tc.name << "\n";
            ++failed;
        }
        else
        {
            ++passed;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "\n========================================\n";
    std::cout << "  " << tests.size() << " tests | "
              << passed << " passed | "
              << failed << " failed | "
              << ms << " ms\n";
    std::cout << "========================================\n";

    passCount() = passed;
    failCount() = failed;

    return failed > 0 ? 1 : 0;
}

} // namespace dspark::test

// ============================================================================
// Macros
// ============================================================================

#define DSPARK_TEST(name)                                                        \
    static void dspark_test_##name();                                            \
    static dspark::test::AutoRegister dspark_reg_##name(#name, dspark_test_##name); \
    static void dspark_test_##name()

// -- General assertions -------------------------------------------------------

#define EXPECT_TRUE(expr)                                                        \
    do {                                                                          \
        if (!(expr)) {                                                            \
            std::cerr << "    EXPECT_TRUE failed: " #expr                         \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            dspark::test::currentTestFailed() = true;                             \
            return;                                                               \
        }                                                                         \
    } while (0)

#define EXPECT_FALSE(expr)                                                       \
    do {                                                                          \
        if ((expr)) {                                                             \
            std::cerr << "    EXPECT_FALSE failed: " #expr                        \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            dspark::test::currentTestFailed() = true;                             \
            return;                                                               \
        }                                                                         \
    } while (0)

#define EXPECT_EQ(a, b)                                                          \
    do {                                                                          \
        if ((a) != (b)) {                                                         \
            std::cerr << "    EXPECT_EQ failed: " #a " == " #b                    \
                      << "  (got " << (a) << " vs " << (b) << ")"                 \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            dspark::test::currentTestFailed() = true;                             \
            return;                                                               \
        }                                                                         \
    } while (0)

#define EXPECT_NE(a, b)                                                          \
    do {                                                                          \
        if ((a) == (b)) {                                                         \
            std::cerr << "    EXPECT_NE failed: " #a " != " #b                    \
                      << "  (both are " << (a) << ")"                             \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            dspark::test::currentTestFailed() = true;                             \
            return;                                                               \
        }                                                                         \
    } while (0)

#define EXPECT_NEAR(a, b, tol)                                                   \
    do {                                                                          \
        auto _a = (a); auto _b = (b);                                             \
        if (std::abs(_a - _b) > (tol)) {                                          \
            std::cerr << "    EXPECT_NEAR failed: |" #a " - " #b "| <= " #tol     \
                      << "  (got " << _a << " vs " << _b                          \
                      << ", diff=" << std::abs(_a - _b) << ")"                    \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            dspark::test::currentTestFailed() = true;                             \
            return;                                                               \
        }                                                                         \
    } while (0)

#define EXPECT_GT(a, b)                                                          \
    do {                                                                          \
        if (!((a) > (b))) {                                                       \
            std::cerr << "    EXPECT_GT failed: " #a " > " #b                     \
                      << "  (got " << (a) << " vs " << (b) << ")"                 \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            dspark::test::currentTestFailed() = true;                             \
            return;                                                               \
        }                                                                         \
    } while (0)

#define EXPECT_LT(a, b)                                                          \
    do {                                                                          \
        if (!((a) < (b))) {                                                       \
            std::cerr << "    EXPECT_LT failed: " #a " < " #b                     \
                      << "  (got " << (a) << " vs " << (b) << ")"                 \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            dspark::test::currentTestFailed() = true;                             \
            return;                                                               \
        }                                                                         \
    } while (0)

// -- DSP-specific assertions --------------------------------------------------

#define EXPECT_SILENT(buf, n, threshold)                                         \
    do {                                                                          \
        for (int _i = 0; _i < (n); ++_i) {                                       \
            if (std::abs((buf)[_i]) > (threshold)) {                              \
                std::cerr << "    EXPECT_SILENT failed at sample " << _i          \
                          << ": |" << (buf)[_i] << "| > " << (threshold)          \
                          << "  (" << __FILE__ << ":" << __LINE__ << ")\n";       \
                dspark::test::currentTestFailed() = true;                         \
                return;                                                           \
            }                                                                     \
        }                                                                         \
    } while (0)

#define EXPECT_BOUNDED(buf, n, lo, hi)                                           \
    do {                                                                          \
        for (int _i = 0; _i < (n); ++_i) {                                       \
            if ((buf)[_i] < (lo) || (buf)[_i] > (hi)) {                           \
                std::cerr << "    EXPECT_BOUNDED failed at sample " << _i         \
                          << ": " << (buf)[_i] << " not in [" << (lo)             \
                          << ", " << (hi) << "]"                                  \
                          << "  (" << __FILE__ << ":" << __LINE__ << ")\n";       \
                dspark::test::currentTestFailed() = true;                         \
                return;                                                           \
            }                                                                     \
        }                                                                         \
    } while (0)

#define EXPECT_NO_NAN(buf, n)                                                    \
    do {                                                                          \
        for (int _i = 0; _i < (n); ++_i) {                                       \
            if (std::isnan((buf)[_i]) || std::isinf((buf)[_i])) {                 \
                std::cerr << "    EXPECT_NO_NAN failed at sample " << _i          \
                          << ": " << (buf)[_i]                                    \
                          << "  (" << __FILE__ << ":" << __LINE__ << ")\n";       \
                dspark::test::currentTestFailed() = true;                         \
                return;                                                           \
            }                                                                     \
        }                                                                         \
    } while (0)
