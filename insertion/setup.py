from distutils.core import setup
import numpy
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


setup(
    name = "poission_disk",
    cmdclass = {'build_ext': build_ext},
    ext_modules =cythonize(Extension(name="poission_disk",
              sources=["poission_disk.pyx"],

            include_dirs = [numpy.get_include() ],

              extra_compile_args = [ '-std=c++11',
                                    "-O3",],
              extra_link_args=[],
              language="c++",
              ))
)
