from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np


NAME = "sent2vec"
VERSION = "0.1"
DESCR = "sent2vec as implemented by celento"
URL = "https://github.com/celento/sent2vec"
REQUIRES = ['numpy', 'cython']
AUTHOR = "celento"
SRC_DIR = "src"
PACKAGES = [SRC_DIR]

sources = ['sent2vec.pyx',
           'fasttext.cc',
           'args.cc',
           'dictionary.cc',
           'matrix.cc',
           'qmatrix.cc',
           'model.cc',
           'real.cc',
           'utils.cc',
           'vector.cc',
           'real.cc',
           'productquantizer.cc']

ext_1 = Extension(name="sent2vec",
                  sources=[f"{SRC_DIR}/{source}" for source in sources],
                  include_dirs=[np.get_include()],
                  libraries=[],
                  extra_compile_args=['-std=c++0x', '-pthread', '-w'],  # , '-Wno-sign-compare' '-Wno-cpp'
                  language='c++',
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])

exts = [ext_1]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          url=URL,
          cmdclass={"build_ext": build_ext},
          ext_modules=exts)
