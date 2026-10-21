import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import setuptools

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user
    def __str__(self):
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'capri_cpp',
        sources=[
            'src/bindings.cpp',
            'src/brackets.cpp',
            'src/six_point.cpp',
            'src/conditioning.cpp',
            'src/greedy.cpp',
            'src/stratification.cpp',
            'src/grassmann_plucker.cpp',
            'src/residual.cpp',
            'src/pencil.cpp'
        ],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            'include',
            'include/Eigen'
        ],
        language='c++'
    ),
]

def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    flags = ['-std=c++17']
    for flag in flags:
        if has_flag(compiler, flag): return flag
    raise RuntimeError('Unsupported compiler -- at least C++17 support is needed!')

class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/EHsc', '/std:c++17', '/O2'],
        'unix': ['-O3']  # -ffast-math removed: it breaks the inf/NaN checks used for degenerate tuples
    }
    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='capri_cpp',
    version='1.0.0',
    description='C++ Core for CAPRI',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
