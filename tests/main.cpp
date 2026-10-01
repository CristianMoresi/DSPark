// DSPark Test Suite - Entry point

#include "dspark_test.h"

#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 1 && (argc != 3 || std::string(argv[1]) != "--prefix" || argv[2][0] == '\0'))
    {
        std::cerr << "Usage: dspark_tests [--prefix test_name_prefix]\n";
        return 1;
    }
    std::cout << "DSPark Test Suite\n";
    std::cout << "========================================\n\n";

    return dspark::test::runAll(argc == 3 ? argv[2] : "");
}
