// DSPark Test Suite - Entry point

#include "dspark_test.h"

#include <iostream>

int main()
{
    std::cout << "DSPark Test Suite\n";
    std::cout << "========================================\n\n";

    return dspark::test::runAll();
}
