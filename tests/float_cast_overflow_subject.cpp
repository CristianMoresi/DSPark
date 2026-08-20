// DSPark sanitizer negative control: this program must terminate nonzero when
// float-cast-overflow instrumentation is active.

#include <limits>

int main()
{
    volatile double finiteOutOfRange = std::numeric_limits<double>::max();
    volatile int converted = static_cast<int>(finiteOutOfRange);
    return converted == 0 ? 0 : 0;
}
