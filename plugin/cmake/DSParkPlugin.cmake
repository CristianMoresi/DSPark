# DSPark — dspark_add_plugin(): build a plugin bundle from one source file.
#
#   include(plugin/cmake/DSParkPlugin.cmake)
#   dspark_add_plugin(MySaturator
#       SOURCES   mysaturator.cpp
#       FORMATS   VST3            # CLAP / AU arrive with their backends
#   )
#
# Produces the platform-correct VST3 bundle folder:
#   Windows : MySaturator.vst3/Contents/x86_64-win/MySaturator.vst3
#   Linux   : MySaturator.vst3/Contents/<arch>-linux/MySaturator.so
#   macOS   : MySaturator.vst3/Contents/MacOS/MySaturator + Info.plist
#
# DSPARK_DIR defaults to the directory two levels above this file.

if(NOT DEFINED DSPARK_DIR)
    get_filename_component(DSPARK_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
endif()

function(dspark_add_plugin TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;FORMATS" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "dspark_add_plugin(${TARGET}): SOURCES is required")
    endif()
    if(NOT ARG_FORMATS)
        set(ARG_FORMATS VST3)
    endif()

    set(want_clap FALSE)
    foreach(fmt IN LISTS ARG_FORMATS)
        if(fmt STREQUAL "CLAP")
            set(want_clap TRUE)
        elseif(NOT fmt STREQUAL "VST3")
            message(WARNING "dspark_add_plugin: format ${fmt} not available yet (VST3/CLAP)")
        endif()
    endforeach()

    add_library(${TARGET} MODULE ${ARG_SOURCES})
    target_include_directories(${TARGET} PRIVATE "${DSPARK_DIR}")
    target_compile_features(${TARGET} PRIVATE cxx_std_20)
    set_target_properties(${TARGET} PROPERTIES
        PREFIX ""
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

    if(WIN32)
        set(arch_dir "x86_64-win")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
            set(arch_dir "arm64-win")
        endif()
        set_target_properties(${TARGET} PROPERTIES
            SUFFIX ".vst3"
            LIBRARY_OUTPUT_DIRECTORY
                "${CMAKE_BINARY_DIR}/${TARGET}.vst3/Contents/${arch_dir}")
    elseif(APPLE)
        set_target_properties(${TARGET} PROPERTIES
            SUFFIX ""
            LIBRARY_OUTPUT_DIRECTORY
                "${CMAKE_BINARY_DIR}/${TARGET}.vst3/Contents/MacOS")
        # Minimal bundle metadata; codesign/notarisation is a release step.
        file(WRITE "${CMAKE_BINARY_DIR}/${TARGET}.vst3/Contents/Info.plist"
"<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
  <key>CFBundleExecutable</key><string>${TARGET}</string>
  <key>CFBundleIdentifier</key><string>com.dspark.${TARGET}</string>
  <key>CFBundleName</key><string>${TARGET}</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleVersion</key><string>1.0.0</string>
</dict></plist>
")
    else()
        string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" arch_lower)
        if(arch_lower MATCHES "aarch64|arm64")
            set(arch_dir "aarch64-linux")
        else()
            set(arch_dir "x86_64-linux")
        endif()
        set_target_properties(${TARGET} PROPERTIES
            SUFFIX ".so"
            LIBRARY_OUTPUT_DIRECTORY
                "${CMAKE_BINARY_DIR}/${TARGET}.vst3/Contents/${arch_dir}")
    endif()

    # The same binary carries both entry points when the source uses both
    # macros, so the CLAP "build" is a post-build copy with the right name.
    if(want_clap)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                "$<TARGET_FILE:${TARGET}>"
                "${CMAKE_BINARY_DIR}/${TARGET}.clap"
            COMMENT "dspark_add_plugin: ${TARGET}.clap")
    endif()
endfunction()
