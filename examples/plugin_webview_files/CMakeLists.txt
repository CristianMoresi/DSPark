cmake_minimum_required(VERSION 3.21)
project(DSParkWebGainExample CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../plugin/cmake/DSParkPlugin.cmake")

dspark_add_plugin(DSParkWebGain
    SOURCES         webgain.cpp
    FORMATS         VST3 CLAP AU
    EDITOR_HTML     ui/editor.html
    AU_SUBTYPE      DSwg
    AU_MANUFACTURER DSpk
    AU_NAME         "DSPark: WebGain"
)
