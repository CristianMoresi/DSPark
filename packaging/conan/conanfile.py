# DSPark Conan recipe - submit to conan-center-index once a release tag exists.
from conan import ConanFile
from conan.tools.files import copy, get
from conan.tools.layout import basic_layout
import os


class DSParkConan(ConanFile):
    name = "dspark"
    version = "1.8.0"
    description = ("Header-only audio DSP framework in pure C++20 with zero "
                   "external dependencies: filters, dynamics, reverbs, physical "
                   "analog models, pitch tools, EBU R128 metering and more.")
    license = "MIT"
    url = "https://github.com/conan-io/conan-center-index"
    homepage = "https://github.com/CristianMoresi/DSPark"
    topics = ("audio", "dsp", "header-only", "filters", "effects", "loudness")
    package_type = "header-library"
    settings = "os", "arch", "compiler", "build_type"
    no_copy_source = True

    def layout(self):
        basic_layout(self, src_folder="src")

    def source(self):
        get(self,
            "https://codeload.github.com/CristianMoresi/DSPark/tar.gz/d8a98a6cf3a7c88af7e57a442f34b88fe869885a",
            filename="dspark-1.8.0.tar.gz",
            sha256="e01c8918b8d8293f0b310dcf5a81b47609e4ad8056a8236f1b7c75a941c90f00", strip_root=True)

    def package(self):
        copy(self, "LICENSE", self.source_folder,
             os.path.join(self.package_folder, "licenses"))
        # The same layout as the CMake install: include/dspark is the include
        # root, so #include <DSPark.h> works with every package manager.
        for module in ("Core", "Effects", "Analysis", "IO", "Music"):
            copy(self, "*.h",
                 os.path.join(self.source_folder, module),
                 os.path.join(self.package_folder, "include", "dspark", module))
        copy(self, "DSPark.h", self.source_folder,
             os.path.join(self.package_folder, "include", "dspark"))
        # The plugin layer: format wrappers, vendored format SDK headers with
        # their licenses, the WebView editor and the dspark_add_plugin() helper.
        for pattern in ("*.h", "*.cmake", "LICENSE*"):
            copy(self, pattern,
                 os.path.join(self.source_folder, "plugin"),
                 os.path.join(self.package_folder, "include", "dspark", "plugin"))

    def package_info(self):
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        self.cpp_info.includedirs = ["include/dspark"]
        self.cpp_info.builddirs = ["include/dspark/plugin/cmake"]
        self.cpp_info.set_property("cmake_file_name", "dspark")
        self.cpp_info.set_property("cmake_target_name", "dspark::dspark")
        # CMakeDeps loads it with the package: dspark_add_plugin() and
        # dspark_embed_editor(), rooted at the packaged headers.
        self.cpp_info.set_property(
            "cmake_build_modules", ["include/dspark/plugin/cmake/DSParkPlugin.cmake"])

    def package_id(self):
        self.info.clear()
