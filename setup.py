import multiprocessing
import sys
import subprocess
from distutils.command.build import build
from setuptools import setup
import os
from distutils import sysconfig

python_version = str(sys.version_info[0])
package_path = 'bin/python' + python_version + '/pip_package'
package_assembly_script = 'scripts/assemble_pip_package.sh'
supported_platforms = ["Linux", "Mac OS-X"]

if sys.platform.startswith("win"):
    raise RuntimeError("Building pip package on Windows is not currently available ...")
elif sys.platform.startswith("darwin"):
    dynamic_library_extension = "dylib"
elif sys.platform.startswith("linux"):
    dynamic_library_extension = "so"
else:
    raise RuntimeError("Unrecognized platform: {}".format(sys.platform))

# TODO: make it run only when install/build is issued
subprocess.check_call(['mkdir', '-p', package_path])


class BuildCommand(build):
    def run(self):
        try:
            cpu_cores = max(1, multiprocessing.cpu_count() - 1)
            python_library = os.path.join(sysconfig.get_config_var('LIBPL'),
                                          'libpython{}.{}'.format(sysconfig.get_python_version(),
                                                                  dynamic_library_extension))
            python_include_dir = sysconfig.get_python_inc()
            cmake_arg_list = list()
            cmake_arg_list.append("-DCMAKE_BUILD_TYPE=Release")
            cmake_arg_list.append("-DBUILD_PYTHON=ON")
            cmake_arg_list.append('-DPYTHON_LIBRARY={}'.format(python_library))
            cmake_arg_list.append('-DPYTHON_INCLUDE_DIR={}'.format(python_include_dir))
            if python_version == "3":
                cmake_arg_list.append("-DBUILD_PYTHON3=ON")
            else:
                cmake_arg_list.append("-DBUILD_PYTHON3=OFF")

            subprocess.check_call(['rm', '-f', 'CMakeCache.txt'])
            subprocess.check_call(['cmake'] + cmake_arg_list)
            subprocess.check_call(['make', '-j', str(cpu_cores)])
        except subprocess.CalledProcessError:
            sys.stderr.write(
                "\033[1m" + "\nInstallation failed, you may be missing some dependencies. "
                            "\nPlease check https://github.com/Marqt/ViZDoom/blob/master/doc/Building.md "
                            "for details\n\n"
                + "\033[0m")
            raise
        build.run(self)


setup(
    name='vizdoom',
    version='1.1.0',
    description='Reinforcement learning platform based on Doom',
    long_description="ViZDoom allows developing AI bots that play Doom using only the visual information (the screen buffer). " \
                     "It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.",
    url='http://vizdoom.cs.put.edu.pl/',
    author='ViZDoom Team',
    author_email='vizdoom@googlegroups.com',

    install_requires=['numpy'],
    setup_requires=['numpy'],
    packages=['vizdoom'],
    package_dir={'vizdoom': package_path},
    package_data={'vizdoom': ['freedoom2.wad', 'vizdoom', 'vizdoom.pk3', 'vizdoom.so', 'bots.cfg', 'scenarios/*']},
    include_package_data=True,
    cmdclass={'build': BuildCommand},
    platforms=supported_platforms,
    classifiers=[
        # Development Status :: 1 - Planning
        # Development Status :: 2 - Pre-Alpha
        # Development Status :: 3 - Alpha
        # Development Status :: 4 - Beta
        'Development Status :: 5 - Production/Stable',
        # Development Status :: 6 - Mature
        # Development Status :: 7 - Inactive
        # 'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords=['vizdoom', 'doom', 'ai', 'deep learning', 'reinforcement learning', 'research']

)
