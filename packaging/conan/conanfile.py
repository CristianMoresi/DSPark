# DSPark Conan 2 recipe for local use; not published to Conan Center.
from conan import ConanFile
from conan.tools.files import copy, get
from conan.tools.layout import basic_layout
import os


class DSParkConan(ConanFile):
    name = "dspark"
    version = "1.8.0"
    description = ("Header-only C++20 audio DSP for real-time and offline processing: "
                   "effects, analog circuit models, automatic dynamics, stereo tools "
                   "and EBU R128 metering. No external DSP dependencies.")
    license = "MIT"
    url = "https://github.com/CristianMoresi/DSPark"
    homepage = "https://github.com/CristianMoresi/DSPark"
    topics = ("audio", "dsp", "header-only", "effects", "loudness", "offline-audio")
    package_type = "header-library"
    settings = "os", "arch", "compiler", "build_type"
    no_copy_source = True

    def layout(self):
        basic_layout(self, src_folder="src")

    def source(self):
        get(self,
            "https://codeload.github.com/CristianMoresi/DSPark/tar.gz/a69cee3d5c2cfd1cfa9c11cfd67891227c1427ae",
            filename="dspark-1.8.0.tar.gz",
            sha256="9bac2beec7f360a308f7cab8a454682eabd61d85c185516a00c52c66fa3447fe", strip_root=True)

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
            "cmake_build_modules",
            [os.path.join("include", "dspark", "plugin", "cmake", "DSParkPlugin.cmake")])

    def package_id(self):
        self.info.clear()
