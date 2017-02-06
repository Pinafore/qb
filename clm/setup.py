from distutils.core import setup, Extension


clm_module = Extension(
    '_clm',
    sources=['clm.cpp'],
    extra_compile_args=['-std=c++11']
)

setup(
    name='clm',
    version='1.0',
    ext_modules=[clm_module],
    py_modules=['clm']
)
