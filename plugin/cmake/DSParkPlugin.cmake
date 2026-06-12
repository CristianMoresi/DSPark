# DSPark — dspark_add_plugin(): build a plugin bundle from one source file.
#
#   include(plugin/cmake/DSParkPlugin.cmake)
#   dspark_add_plugin(MySaturator
#       SOURCES         mysaturator.cpp
#       FORMATS         VST3 CLAP AU        # AU builds on macOS only
#       EDITOR_HTML     ui/editor.html      # optional: WebView editor page
#       AU_SUBTYPE      Subt                # required with AU: unique 4-char
#       AU_MANUFACTURER Manu                # required with AU: unique 4-char
#   )
#
# Produces the platform-correct bundle layouts:
#   Windows : MySaturator.vst3/Contents/x86_64-win/MySaturator.vst3
#   Linux   : MySaturator.vst3/Contents/<arch>-linux/MySaturator.so
#   macOS   : MySaturator.vst3/Contents/MacOS/MySaturator + Info.plist
#   CLAP    : MySaturator.clap (same binary, copied)
#   AU      : MySaturator.component/Contents/{MacOS,Info.plist} (macOS)
#
# EDITOR_HTML embeds the page at build time (see dspark_embed_editor below):
# develop the UI as ordinary separate .html/.css/.js files; the build inlines
# them into a generated header. Include it from the plugin source as
#   #include "<Target>_editor_html.h"        // defines kDsparkEditorHtml
#   static const char* editorHtml() { return kDsparkEditorHtml; }
#
# DSPARK_DIR defaults to the directory two levels above this file.

if(NOT DEFINED DSPARK_DIR)
    get_filename_component(DSPARK_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
endif()
set(_DSPARK_EMBED_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/DSParkEmbedEditor.cmake")

# -- dspark_embed_editor ----------------------------------------------------------
#
#   dspark_embed_editor(<target>
#       HTML <page.html>            # the editor page (may reference local
#                                   # stylesheets/scripts; they get inlined)
#       [HEADER <name.h>]           # generated header (default <target>_editor_html.h)
#       [VARIABLE <identifier>]     # C++ array name (default kDsparkEditorHtml)
#   )
#
# Regenerates the header whenever the page OR any file it references changes,
# and adds the generated directory to the target's include path.

function(dspark_embed_editor TARGET)
    cmake_parse_arguments(EMB "" "HTML;HEADER;VARIABLE" "" ${ARGN})
    if(NOT EMB_HTML)
        message(FATAL_ERROR "dspark_embed_editor(${TARGET}): HTML is required")
    endif()
    if(NOT EMB_HEADER)
        set(EMB_HEADER "${TARGET}_editor_html.h")
    endif()
    if(NOT EMB_VARIABLE)
        set(EMB_VARIABLE "kDsparkEditorHtml")
    endif()
    get_filename_component(page "${EMB_HTML}" ABSOLUTE
                           BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${page}")
        message(FATAL_ERROR "dspark_embed_editor(${TARGET}): ${page} not found")
    endif()

    # Discover the referenced local assets so edits to them retrigger the
    # embed (the script re-resolves them at build time anyway).
    get_filename_component(page_dir "${page}" DIRECTORY)
    set(assets "")
    file(READ "${page}" page_text)
    string(REGEX MATCHALL "(href|src)=\"([^\"]+)\"" refs "${page_text}")
    foreach(ref IN LISTS refs)
        string(REGEX REPLACE "^(href|src)=\"([^\"]+)\"$" "\\2" path "${ref}")
        if(NOT path MATCHES "^[a-zA-Z][a-zA-Z0-9+.-]*:" AND EXISTS "${page_dir}/${path}")
            list(APPEND assets "${page_dir}/${path}")
        endif()
    endforeach()

    set(header "${CMAKE_CURRENT_BINARY_DIR}/${EMB_HEADER}")
    add_custom_command(OUTPUT "${header}"
        COMMAND ${CMAKE_COMMAND}
            -DINPUT=${page} -DOUTPUT=${header} -DVARIABLE=${EMB_VARIABLE}
            -P "${_DSPARK_EMBED_SCRIPT}"
        DEPENDS "${page}" ${assets} "${_DSPARK_EMBED_SCRIPT}"
        COMMENT "dspark_embed_editor: ${EMB_HEADER}")
    target_sources(${TARGET} PRIVATE "${header}")
    target_include_directories(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
endfunction()

# -- dspark_add_plugin --------------------------------------------------------------

function(dspark_add_plugin TARGET)
    cmake_parse_arguments(ARG ""
        "EDITOR_HTML;AU_TYPE;AU_SUBTYPE;AU_MANUFACTURER;AU_NAME"
        "SOURCES;FORMATS" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "dspark_add_plugin(${TARGET}): SOURCES is required")
    endif()
    if(NOT ARG_FORMATS)
        set(ARG_FORMATS VST3)
    endif()

    set(want_clap FALSE)
    set(want_au FALSE)
    foreach(fmt IN LISTS ARG_FORMATS)
        if(fmt STREQUAL "CLAP")
            set(want_clap TRUE)
        elseif(fmt STREQUAL "AU")
            set(want_au TRUE)
        elseif(NOT fmt STREQUAL "VST3")
            message(WARNING "dspark_add_plugin: unknown format ${fmt} "
                            "(VST3/CLAP/AU)")
        endif()
    endforeach()

    add_library(${TARGET} MODULE ${ARG_SOURCES})
    target_include_directories(${TARGET} PRIVATE "${DSPARK_DIR}")
    target_compile_features(${TARGET} PRIVATE cxx_std_20)
    set_target_properties(${TARGET} PROPERTIES
        PREFIX ""
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

    if(ARG_EDITOR_HTML)
        dspark_embed_editor(${TARGET} HTML "${ARG_EDITOR_HTML}")
    endif()

    if(WIN32)
        # System libs the WebView editor layer uses (MSVC auto-links them via
        # #pragma comment; this covers MinGW/clang). Harmless without an editor.
        target_link_libraries(${TARGET} PRIVATE
            advapi32 comctl32 ole32 shell32 shlwapi user32 version)
        set(arch_dir "x86_64-win")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
            set(arch_dir "arm64-win")
        endif()
        set_target_properties(${TARGET} PROPERTIES
            SUFFIX ".vst3"
            # $<1:...> keeps multi-config generators (Visual Studio) from
            # appending a per-config subdirectory inside the bundle.
            LIBRARY_OUTPUT_DIRECTORY
                "$<1:${CMAKE_BINARY_DIR}/${TARGET}.vst3/Contents/${arch_dir}>")
    elseif(APPLE)
        # objc runtime for the WKWebView editor glue, and the system audio
        # frameworks the AU entry points reference; harmless when unused.
        target_link_libraries(${TARGET} PRIVATE objc
            "-framework AudioToolbox" "-framework CoreFoundation")
        set_target_properties(${TARGET} PROPERTIES
            SUFFIX ""
            LIBRARY_OUTPUT_DIRECTORY
                "$<1:${CMAKE_BINARY_DIR}/${TARGET}.vst3/Contents/MacOS>")
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
                "$<1:${CMAKE_BINARY_DIR}/${TARGET}.vst3/Contents/${arch_dir}>")
    endif()

    # The same binary carries every entry point the source declares, so the
    # CLAP "build" is a post-build copy with the right name (a bundle folder
    # on macOS, where hosts require it; a plain file elsewhere).
    if(want_clap)
        if(APPLE)
            set(clap_bundle "${CMAKE_BINARY_DIR}/${TARGET}.clap")
            file(WRITE "${clap_bundle}/Contents/Info.plist"
"<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
  <key>CFBundleExecutable</key><string>${TARGET}</string>
  <key>CFBundleIdentifier</key><string>com.dspark.${TARGET}.clap</string>
  <key>CFBundleName</key><string>${TARGET}</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleVersion</key><string>1.0.0</string>
</dict></plist>
")
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                    "$<TARGET_FILE:${TARGET}>"
                    "${clap_bundle}/Contents/MacOS/${TARGET}"
                COMMAND codesign --force -s - "${clap_bundle}"
                COMMENT "dspark_add_plugin: ${TARGET}.clap (bundle)")
        else()
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                    "$<TARGET_FILE:${TARGET}>"
                    "${CMAKE_BINARY_DIR}/${TARGET}.clap"
                COMMENT "dspark_add_plugin: ${TARGET}.clap")
        endif()
    endif()

    # AU: a .component bundle around the same binary, with the AudioComponents
    # entry auval/Logic discover (DSPARK_AU_PLUGIN exports DSParkAuFactory).
    if(want_au)
        if(NOT APPLE)
            message(STATUS "dspark_add_plugin: AU skipped (macOS only)")
            return()
        endif()
        if(NOT ARG_AU_SUBTYPE OR NOT ARG_AU_MANUFACTURER)
            message(FATAL_ERROR "dspark_add_plugin(${TARGET}): AU requires "
                                "AU_SUBTYPE and AU_MANUFACTURER (4 chars each, "
                                "matching the DSPARK_AU_PLUGIN macro)")
        endif()
        if(NOT ARG_AU_TYPE)
            set(ARG_AU_TYPE "aufx")
        endif()
        if(NOT ARG_AU_NAME)
            set(ARG_AU_NAME "${TARGET}")
        endif()
        set(component "${CMAKE_BINARY_DIR}/${TARGET}.component")
        file(WRITE "${component}/Contents/Info.plist"
"<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
  <key>CFBundleExecutable</key>
  <string>${TARGET}</string>
  <key>CFBundleIdentifier</key>
  <string>com.dspark.${TARGET}.au</string>
  <key>CFBundleName</key>
  <string>${TARGET}</string>
  <key>CFBundlePackageType</key>
  <string>BNDL</string>
  <key>CFBundleVersion</key>
  <string>1.0.0</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0.0</string>
  <key>AudioComponents</key>
  <array>
    <dict>
      <key>description</key>
      <string>${ARG_AU_NAME}</string>
      <key>factoryFunction</key>
      <string>DSParkAuFactory</string>
      <key>manufacturer</key>
      <string>${ARG_AU_MANUFACTURER}</string>
      <key>name</key>
      <string>${ARG_AU_NAME}</string>
      <key>subtype</key>
      <string>${ARG_AU_SUBTYPE}</string>
      <key>type</key>
      <string>${ARG_AU_TYPE}</string>
      <key>version</key>
      <integer>65536</integer>
    </dict>
  </array>
</dict>
</plist>
")
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                "$<TARGET_FILE:${TARGET}>"
                "${component}/Contents/MacOS/${TARGET}"
            COMMAND codesign --force -s - "${component}"
            COMMENT "dspark_add_plugin: ${TARGET}.component (+ ad-hoc codesign)")
    endif()
endfunction()
