import multiprocessing
import os
import sys
import subprocess
from distutils import sysconfig
from distutils.command.build import build
from setuptools import setup, find_packages

python_version = str(sys.version_info[0])
package_path = 'bin/python' + python_version + '/vizdoom'


def build_task(cmake_arg_list=None):
    # TODO overrice pythonlibs and python interpreter

    cpu_cores = max(1, multiprocessing.cpu_count() - 1)

    cmake_arg_list = cmake_arg_list if cmake_arg_list is not None else []
    cmake_arg_list.append("-DCMAKE_BUILD_TYPE=Release")
    cmake_arg_list.append("-DBUILD_PYTHON=ON")
    if python_version == "3":
        cmake_arg_list.append("-DBUILD_PYTHON3=ON")
    else:
        cmake_arg_list.append("-DBUILD_PYTHON3=OFF")

    subprocess.check_call(['rm', '-f', 'CMakeCache.txt'])
    subprocess.check_call(['cmake'] + cmake_arg_list)
    subprocess.check_call(['make', '-j', str(cpu_cores)])


if sys.platform.startswith("darwin"):
    build_func = build_task
elif sys.platform.startswith("linux"):
    build_func = build_task
elif sys.platform.startswith("win"):
    raise RuntimeError("Building pip package on Windows is not currently available ...")
else:
    raise RuntimeError("Unrecognized platform: {}".format(sys.platform))


class BuildDoom(build):
    def run(self):
        try:
            build_func()
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                "\033[1m" + "\nTODO installation failed, see our page ...\n\n"
                + "\033[0m")
            raise
        build.run(self)


setup(
    name='vizdoom',
    version='1.2.0-dev',
    description='ViZDoom Environment',
    long_description="ViZDoom: Doom Reinforcement Learning Research Platform",
    url='http://vizdoom.cs.put.edu.pl/',
    author='TODO',
    author_email='TODO',
    license='MIT',

    install_requires=['numpy'],
    setup_requires=['numpy'],
    packages=['vizdoom'],
    package_dir={package_path},
    package_data={'vizdoom': ['freedoom2.wad', 'vizdoom', 'vizdoom.pk3', 'vizdoom.so', 'bots.cfg']},
    cmdclass={'build': BuildDoom},

    classifiers=[
        # Development Status :: 1 - Planning
        # Development Status :: 2 - Pre-Alpha
        # Development Status :: 3 - Alpha
        # Development Status :: 4 - Beta
        # Development Status :: 5 - Production/Stable
        # Development Status :: 6 - Mature
        # Development Status :: 7 - Inactive
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='doom ai deep_learning reinforcement_learning research'

)
