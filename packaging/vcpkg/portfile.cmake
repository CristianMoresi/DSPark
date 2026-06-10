# DSPark vcpkg port — submit to microsoft/vcpkg once a release tag exists.
# Update REF and SHA512 for the published release archive
# (vcpkg hashes the GitHub source tarball of the tag).

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO CristianMoresi/DSPark
    REF v1.2.0
    SHA512 0   # placeholder: vcpkg prints the real hash on first install attempt
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DDSPARK_BUILD_CONFORMANCE=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME DSPark CONFIG_PATH share/DSPark)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
