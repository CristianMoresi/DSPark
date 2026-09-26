# DSPark - installed-package consumer (CI): proves that find_package(dspark)
# delivers the library target AND the plugin layer with dspark_add_plugin().
#
#   cmake -S . -B build -DCMAKE_INSTALL_PREFIX=prefix && cmake --install build
#   cmake -S tests/package_consumer -B consumer -DCMAKE_PREFIX_PATH=prefix
#   cmake --build consumer && ./consumer/consumer_app

cmake_minimum_required(VERSION 3.21)
project(DSParkPackageConsumer CXX)

find_package(dspark CONFIG REQUIRED)

add_executable(consumer_app app.cpp)
target_link_libraries(consumer_app PRIVATE dspark::dspark)

dspark_add_plugin(ConsumerPlugin
    SOURCES   plugin.cpp
    FORMATS   VST3 CLAP
    VERSION   2.1.3
    BUNDLE_ID com.dspark.test.consumer)
