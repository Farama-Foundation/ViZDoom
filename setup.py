import sys, os, subprocess, sysconfig
from distutils import sysconfig
from distutils.command.build import build
from multiprocessing import cpu_count
from setuptools import setup


platform = sys.platform
python_version = sysconfig.get_python_version()
package_path = 'bin/python' + python_version + '/pip_package'
supported_platforms = ["Linux", "Mac OS-X"]

if platform.startswith("win"):
    raise RuntimeError("Building pip package on Windows is not currently available ...")
elif platform.startswith("darwin"):
    library_extension = "dylib"
elif platform.startswith("linux"):
    library_extension = "so"
else:
    raise RuntimeError("Unrecognized platform: {}".format(sys.platform))

subprocess.check_call(['mkdir', '-p', package_path])


def get_vizdoom_version():
    try:
        import re
        with open("CMakeLists.txt") as cmake_file:
            lines = "".join(cmake_file.readlines())
            major_version = re.search('set\(ViZDoom_MAJOR_VERSION\s+([0-9])\)\s+', lines).group(1)
            minor_version = re.search('set\(ViZDoom_MINOR_VERSION\s+([1-9]?[0-9])\)\s+', lines).group(1)
            patch_version = re.search('set\(ViZDoom_PATCH_VERSION\s+([1-9]?[0-9]+)\)\s+', lines).group(1)

            version = "{}.{}.{}".format(major_version, minor_version, patch_version)
            return version

    except Exception:
        raise RuntimeError("Package version retrieval failed. "
                           "Most probably something is wrong with this code and "
                           "you should create an issue at https://github.com/mwydmuch/ViZDoom/")


def get_python_library(python_lib_dir):
    if python_version[0] == 2:
        python_lib_name = 'libpython{}.{}'
    else:
        python_lib_name = 'libpython{}m.{}'

    python_lib_name = python_lib_name.format(python_version, library_extension)
    python_library = os.path.join(python_lib_dir, python_lib_name)
    return python_library


class BuildCommand(build):
    def run(self):
        try:
            cpu_cores = max(1, cpu_count() - 1)
            python_executable = os.path.realpath(sys.executable)

            cmake_arg_list = list()
            cmake_arg_list.append("-DCMAKE_BUILD_TYPE=Release")
            cmake_arg_list.append("-DBUILD_PYTHON=ON")
            cmake_arg_list.append("-DPYTHON_EXECUTABLE={}".format(python_executable))

            python_standard_lib = sysconfig.get_python_lib(standard_lib=True)
            python_site_packages = sysconfig.get_python_lib(standard_lib=False)
            python_lib_dir = os.path.dirname(python_standard_lib)
            python_library = get_python_library(python_lib_dir)
            python_include_dir = sysconfig.get_python_inc()
            numpy_include_dir = os.path.join(python_site_packages, "numpy/core/include")

            if os.path.exists(python_library) and os.path.exists(python_include_dir) and os.path.exists(numpy_include_dir):
                cmake_arg_list.append("-DPYTHON_LIBRARY={}".format(python_library))
                cmake_arg_list.append("-DPYTHON_INCLUDE_DIR={}".format(python_include_dir))
                cmake_arg_list.append("-DNUMPY_INCLUDES={}".format(numpy_include_dir))

            if python_version[0] == "3":
                cmake_arg_list.append("-DBUILD_PYTHON3=ON")

            subprocess.check_call(['rm', '-f', 'CMakeCache.txt'])
            subprocess.check_call(['cmake'] + cmake_arg_list)
            subprocess.check_call(['make', '-j', str(cpu_cores)])
        except subprocess.CalledProcessError:
            sys.stderr.write("\033[1m\nInstallation failed, you may be missing some dependencies. "
                             "\nPlease check https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md "
                             "for details\n\n\033[0m")
            raise
        build.run(self)


setup(
    name='vizdoom',
    version=get_vizdoom_version(),
    description='Reinforcement learning platform based on Doom',
    long_description="ViZDoom allows developing AI bots that play Doom using only the visual information (the screen buffer). " \
                     "It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.",
    url='http://vizdoom.cs.put.edu.pl/',
    author='Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, Grzegorz Runc, Jakub Toczek',
    author_email='vizdoom@googlegroups.com',

    install_requires=['numpy'],
    setup_requires=['numpy'],
    packages=['vizdoom'],
    package_dir={'vizdoom': package_path},
    package_data={'vizdoom': ['__init__.py', 'bots.cfg', 'freedoom2.wad', 'vizdoom', 'vizdoom.pk3', 'vizdoom.so', 'scenarios/*']},
    include_package_data=True,
    cmdclass={'build': BuildCommand},
    platforms=supported_platforms,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords=['vizdoom', 'doom', 'ai', 'deep learning', 'reinforcement learning', 'research']
)
