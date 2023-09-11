import os
import shutil
import subprocess
import sys
import warnings
from distutils import sysconfig
from distutils.command.build import build
from multiprocessing import cpu_count

from setuptools import Distribution, setup
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel


platform = sys.platform
python_version = sysconfig.get_python_version()
build_output_path = "bin"
package_path = build_output_path + "/python" + python_version + "/pip_package"
supported_platforms = ["Linux", "Mac OS X", "Windows"]
package_data = [
    "__init__.py",
    "bots.cfg",
    "freedoom2.wad",
    "vizdoom.pk3",
    "vizdoom",
    "scenarios/*",
    "gym_wrapper/*",
    "gymnasium_wrapper/*",
]

os.makedirs(package_path, exist_ok=True)

if platform.startswith("win"):
    package_data.extend(["vizdoom.exe", "*.pyd", "*.dll"])
    library_extension = "lib"
elif platform.startswith("darwin"):
    package_data.extend(["vizdoom", "*.so"])
    library_extension = "dylib"
elif platform.startswith("linux"):
    package_data.extend(["vizdoom", "*.so"])
    library_extension = "so"
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


def get_vizdoom_version():
    try:
        import re

        with open("CMakeLists.txt") as cmake_file:
            lines = cmake_file.read()
            version = re.search(r"VERSION\s+([0-9].[0-9].[0-9]+)", lines).group(1)
            return version

    except Exception:
        raise RuntimeError(
            "Package version retrieval failed. "
            "Most probably something is wrong with this code and "
            "you should create an issue at https://github.com/Farama-Foundation/ViZDoom"
        )


def get_long_description():
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "README.md"), encoding="utf-8") as readme_file:
            return readme_file.read()

    except Exception:
        raise RuntimeError(
            "Package description retrieval failed. "
            "Most probably something is wrong with this code and "
            "you should create an issue at https://github.com/Farama-Foundation/ViZDoom"
        )


class Wheel(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        # Mark us as not a pure python package
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = bdist_wheel.get_tag(self)
        return python, abi, plat


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


class BuildCommand(build):
    def run(self):
        cpu_cores = max(1, cpu_count() - 1)
        cmake_arg_list = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_PYTHON=ON",
            f"-DBUILD_PYTHON_VERSION={python_version}",
        ]

        env_cmake_args = os.getenv("VIZDOOM_CMAKE_ARGS")
        if env_cmake_args:
            cmake_arg_list += env_cmake_args.split()
            warnings.warn(
                f"VIZDOOM_CMAKE_ARGS is set, the following arguments will be added to cmake command: {env_cmake_args}"
            )

        # Windows specific version of the libraries
        if platform.startswith("win"):
            generator = os.getenv("VIZDOOM_BUILD_GENERATOR_NAME")
            if not generator:
                raise RuntimeError(
                    "VIZDOOM_BUILD_GENERATOR_NAME is not set"
                )  # TODO: Improve

            deps_root = os.getenv("VIZDOOM_WIN_DEPS_ROOT")
            if deps_root is None:
                raise RuntimeError("VIZDOOM_WIN_DEPS_ROOT is not set")  # TODO: Improve

            mpg123_include = os.path.join(deps_root, "libmpg123")
            mpg123_lib = os.path.join(deps_root, "libmpg123/libmpg123-0.lib")
            mpg123_dll = os.path.join(deps_root, "libmpg123/libmpg123-0.dll")

            sndfile_include = os.path.join(deps_root, "libsndfile/include")
            sndfile_lib = os.path.join(deps_root, "libsndfile/lib/libsndfile-1.lib")
            sndfile_dll = os.path.join(deps_root, "libsndfile/bin/libsndfile-1.dll")

            os.environ["OPENALDIR"] = str(os.path.join(deps_root, "openal-soft"))
            openal_dll = os.path.join(deps_root, "openal-soft/bin/Win64/OpenAL32.dll")

            cmake_arg_list.extend(
                [
                    "-G",
                    generator,
                    f"-DMPG123_INCLUDE_DIR={mpg123_include}",
                    f"-DMPG123_LIBRARIES={mpg123_lib}",
                    f"-DSNDFILE_INCLUDE_DIR={sndfile_include}",
                    f"-DSNDFILE_LIBRARY={sndfile_lib}",
                ]
            )

            shutil.copy(mpg123_dll, build_output_path)
            shutil.copy(sndfile_dll, build_output_path)
            shutil.copy(openal_dll, build_output_path)

        # python_standard_lib = sysconfig.get_python_lib(standard_lib=True)
        python_root_dir = os.path.dirname(sys.executable)

        if python_root_dir and os.path.exists(python_root_dir):
            cmake_arg_list.append(f"-DPython_ROOT_DIR={python_root_dir}")

        if os.path.exists("CMakeCache.txt"):
            os.remove("CMakeCache.txt")

        print(f"Running cmake with arguments: {cmake_arg_list}", file=sys.stderr)

        try:
            if platform.startswith("win"):
                if os.path.exists("./src/lib_python/libvizdoom_python.dir"):
                    shutil.rmtree(
                        "./src/lib_python/libvizdoom_python.dir"
                    )  # TODO: This is not very elegant, improve
                subprocess.check_call(cmake_arg_list)
                subprocess.check_call(["cmake", "--build", ".", "--config", "Release"])
            else:
                subprocess.check_call(cmake_arg_list)
                subprocess.check_call(["make", "-j", str(cpu_cores)])
        except subprocess.CalledProcessError:
            sys.stderr.write(
                "\033[1m\nInstallation failed, you may be missing some dependencies. "
                "\nPlease check https://github.com/mwydmuch/ViZDoom/blob/master/doc/Installation.md "
                "for details\n\n\033[0m"
            )
            raise

        build.run(self)


setup(
    name="vizdoom",
    version=get_vizdoom_version(),
    description="ViZDoom is Doom-based AI Research Platform for Reinforcement Learning from Raw Visual Information.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="http://vizdoom.cs.put.edu.pl/",
    author="Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, Grzegorz Runc, Jakub Toczek, and the respective contributors",
    author_email="mwydmuch@cs.put.poznan.pl",
    extras_require={
        "gym": ["gym>=0.26.0", "pygame>=2.1.3"],
        "test": ["pytest", "psutil"],
    },
    install_requires=["numpy", "gymnasium>=0.28.0", "pygame>=2.1.3"],
    packages=["vizdoom"],
    package_dir={"vizdoom": package_path},
    package_data={"vizdoom": package_data},
    include_package_data=True,
    cmdclass={"bdist_wheel": Wheel, "build": BuildCommand, "install": InstallPlatlib},
    distclass=BinaryDistribution,
    platforms=supported_platforms,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    keywords=[
        "vizdoom",
        "doom",
        "ai",
        "deep learning",
        "reinforcement learning",
        "research",
    ],
)
