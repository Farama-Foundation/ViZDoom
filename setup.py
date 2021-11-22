import sys, os, subprocess, shutil
from distutils import sysconfig
from distutils.command.build import build
from multiprocessing import cpu_count
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel

platform = sys.platform
python_version = sysconfig.get_python_version()
build_output_path = 'bin'
package_path = build_output_path + '/python' + python_version + '/pip_package'
supported_platforms = ["Linux", "Mac OS X", "Windows"]
package_data = ['__init__.py', 'bots.cfg', 'freedoom2.wad', 'vizdoom.pk3', 'vizdoom', 'scenarios/*']

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
    raise RuntimeError("Unsupported platform: {}".format(sys.platform))
   

def get_vizdoom_version():
    try:
        import re
        with open("CMakeLists.txt") as cmake_file:
            lines = cmake_file.read()
            version = re.search("VERSION\s+([0-9].[0-9].[0-9]+)", lines).group(1)
            return version

    except Exception:
        raise RuntimeError("Package version retrieval failed. "
                           "Most probably something is wrong with this code and "
                           "you should create an issue at https://github.com/mwydmuch/ViZDoom/")


def get_python_library(python_root_dir):
    paths_to_check = [
        "libs/python{}{}.{}", # Windows Python/Anaconda
        "libpython{}.{}m.{}", # Unix
        "libpython{}.{}.{}", # Unix
        "lib/libpython{}.{}m.{}", # Unix Anaconda
        "lib/libpython{}.{}.{}", # Unix Anaconda
    ]
    
    for path_format in paths_to_check:
        path = os.path.join(python_root_dir, path_format.format(*python_version.split('.'), library_extension))
        if os.path.exists(path):
            return path

    return None
    
    
class Wheel(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        # Mark us as not a pure python package
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = bdist_wheel.get_tag(self)
        return python, abi, plat


class BuildCommand(build):
    def run(self):
        try:

            cpu_cores = max(1, cpu_count() - 1)
            python_executable = os.path.realpath(sys.executable)

            cmake_arg_list = ["cmake", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_PYTHON=ON",
                              "-DPYTHON_EXECUTABLE={}".format(python_executable)]
                              
            if platform.startswith("win"):
                generator = os.getenv('VIZDOOM_BUILD_GENERATOR_NAME')
                if not generator:
                    raise RuntimeError("VIZDOOM_BUILD_GENERATOR_NAME is not set") # TODO: Improve
                    
                deps_root = os.getenv('VIZDOOM_WIN_DEPS_ROOT') 
                if deps_root is None:
                    raise RuntimeError("VIZDOOM_WIN_DEPS_ROOT is not set") # TODO: Improve
                
                mpg123_include=os.path.join(deps_root, 'libmpg123')
                mpg123_lib=os.path.join(deps_root, 'libmpg123/libmpg123-0.lib')
                mpg123_dll=os.path.join(deps_root, 'libmpg123/libmpg123-0.dll')

                sndfile_include=os.path.join(deps_root, 'libsndfile/include')
                sndfile_lib=os.path.join(deps_root, 'libsndfile/lib/libsndfile-1.lib')
                sndfile_dll=os.path.join(deps_root, 'libsndfile/bin/libsndfile-1.dll')
                
                os.environ["OPENALDIR"] = str(os.path.join(deps_root, 'openal-soft'))
                openal_dll=os.path.join(deps_root, 'openal-soft/bin/Win64/OpenAL32.dll')
                
                cmake_arg_list.extend(
                    ["-G",
                     generator,
                     "-DNO_ASM=ON",
                     "-DMPG123_INCLUDE_DIR={}".format(mpg123_include),
                     "-DMPG123_LIBRARIES={}".format(mpg123_lib),
                     "-DSNDFILE_INCLUDE_DIR={}".format(sndfile_include),
                     "-DSNDFILE_LIBRARY={}".format(sndfile_lib)
                    ]
                )
                
                shutil.copy(mpg123_dll, build_output_path)
                shutil.copy(sndfile_dll, build_output_path)
                shutil.copy(openal_dll, build_output_path)

            python_standard_lib = sysconfig.get_python_lib(standard_lib=True)
            python_root_dir = os.path.dirname(python_standard_lib)
            python_library = get_python_library(python_root_dir)
            python_include_dir = sysconfig.get_python_inc()
            
            if python_include_dir and os.path.exists(python_include_dir):
                cmake_arg_list.append("-DPYTHON_INCLUDE_DIR={}".format(python_include_dir))

            if python_library and os.path.exists(python_library):
                cmake_arg_list.append("-DPYTHON_LIBRARY={}".format(python_library))
            
            if os.path.exists('CMakeCache.txt'):
                os.remove('CMakeCache.txt')
            
            if platform.startswith("win"):
                if os.path.exists("./src/lib_python/libvizdoom_python.dir"):
                    shutil.rmtree("./src/lib_python/libvizdoom_python.dir") # TODO: This is not very elegant, improve
                subprocess.check_call(cmake_arg_list)
                subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'])
            else:
                subprocess.check_call(cmake_arg_list)
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
    author_email='mwydmuch@cs.put.poznan.pl',

    install_requires=['numpy'],
    packages=['vizdoom'],
    package_dir={'vizdoom': package_path},
    package_data={'vizdoom': package_data},
    include_package_data=True,
    cmdclass={'bdist_wheel': Wheel, 'build': BuildCommand},
    platforms=supported_platforms,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
    ],
    keywords=['vizdoom', 'doom', 'ai', 'deep learning', 'reinforcement learning', 'research']
)
