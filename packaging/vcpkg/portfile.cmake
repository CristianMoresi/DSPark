# DSPark vcpkg overlay port for local use; not published to the central registry.
# REF pins the immutable source commit.
# SHA512 authenticates the archive for that exact commit.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO CristianMoresi/DSPark
    REF a69cee3d5c2cfd1cfa9c11cfd67891227c1427ae
    SHA512 3d7a4c6c07e50a31074a3e8435b291e30af1de6d0eb6f721ce0d711fbc7cde4b3620e0f7fe150827a7614a30d0f6716aa5d38556b1244f7013baa5d9f951dc6f
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
