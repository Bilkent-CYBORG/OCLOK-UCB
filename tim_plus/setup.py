from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

sources_list = ["timgraph.pyx", "Graph.cpp", "InfGraph.cpp", "TimGraph.cpp", "sfmt/SFMT.c"]

setup(ext_modules=[Extension("pytim", sources=sources_list, language="c++", extra_compile_args=["-std=c++11"])],
      cmdclass={'build_ext': build_ext})
