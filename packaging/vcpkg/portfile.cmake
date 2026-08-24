# DSPark vcpkg port - submit to microsoft/vcpkg once a release tag exists.
# REF pins the immutable source commit.
# SHA512 authenticates the archive for that exact commit.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO CristianMoresi/DSPark
    REF 5a47d959de4b3d48445a8850960f74377999faf9
    SHA512 b7382dc3e0247fbf93e0a5555750deda74798809d4c3d8154a08bb517faf9cf046cabdd04d88fc6e7895a7a6666e7017eeed719ea2582edca5363bba8ef9383f
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DDSPARK_BUILD_CONFORMANCE=OFF
        -DDSPARK_BUILD_TESTS=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME dspark CONFIG_PATH lib/cmake/dspark)

# Header-only: no compiled libraries, and config_fixup moved the cmake files
# from lib/cmake to share/, so lib/ is left empty (vcpkg rejects empty
# installed directories).
file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug"
    "${CURRENT_PACKAGES_DIR}/lib")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
