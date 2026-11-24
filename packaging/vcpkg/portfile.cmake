# DSPark vcpkg overlay port for local use; not published to the central registry.
# REF pins the immutable source commit.
# SHA512 authenticates the archive for that exact commit.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO CristianMoresi/DSPark
    REF d72bc3b03340a2fb8cec1dc5ea0de0ea8717ee8b
    SHA512 84987871f7b2485065cbac4f8a2da7c306547c726c9b4f8104728aaa0fd6b1d32af567a9f53ea43be4c6f45baa4690610110d6ae0a3e5e1b4ad428d80ce77b4c
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
