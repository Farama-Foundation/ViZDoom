from distutils.core import setup, Extension
import numpy.distutils.misc_util


c_ext = Extension("api", 
				  sources =["api.cpp", "Bridge.cpp","ViziaDoomController.cpp"],
				  libraries = ['boost_system', 'boost_thread','pthread','rt'])




setup(
    ext_modules=[c_ext],

	include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
