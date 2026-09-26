# DSPark vcpkg port - submit to microsoft/vcpkg once a release tag exists.
# REF pins the immutable source commit.
# SHA512 authenticates the archive for that exact commit.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO CristianMoresi/DSPark
    REF d8a98a6cf3a7c88af7e57a442f34b88fe869885a
    SHA512 f8f1fff5e561a7405a4704dcedaef8fa935845cc373787bedff69c19b21db44ab50c42ffe2e375c5f6624788c61c42f77f5198471988067312481255e847bbd5
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
