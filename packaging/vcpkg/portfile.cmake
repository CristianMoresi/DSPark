# DSPark vcpkg overlay port for local use; not published to the central registry.
# REF pins the immutable source commit.
# SHA512 authenticates the archive for that exact commit.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO CristianMoresi/DSPark
    REF 816766a14e04804ba3425d789e8bb384866067ea
    SHA512 b8c10dc1b8299025c643a6e7cf62f2a23b5f1bc7636a0692fa58424848c5708b14684a9ba1c93f6d73546bfa19f67a67525a3a1cd5f4bf483d441800d6c5833c
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
