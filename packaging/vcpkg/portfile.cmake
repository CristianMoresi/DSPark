# DSPark vcpkg port — submit to microsoft/vcpkg once a release tag exists.
# Update REF and SHA512 for the published release archive
# (vcpkg hashes the GitHub source tarball of the tag).

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO CristianMoresi/DSPark
    REF v1.2.1
    SHA512 4fe046334bdfd5408d833ac9389b6e8656504d9e92bb14d6a4508c775843798e0448ed1417e8abd0a72fbdb74d28ab50e8ca18c4e9e813cb0628a01975110561
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
