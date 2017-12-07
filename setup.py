import os
from setuptools import setup, find_packages

import Cython
from Cython.Build import cythonize

import numpy as np

Cython.Compiler.Options.annotate = True

root = os.path.abspath(os.path.dirname(__file__))
try:
    long_desc = open(os.path.join(root, 'README.md')).read()
except Exception:
    long_desc = '<Missing README.md>'
    print('Missing README.md')

setup(
    name='evosim',
    version='0.1',
    description='Evolution simulator',
    long_description=long_desc,
    author='Ryan Wallace',
    author_email='ryanwallace@college.harvard.edu',
    url='https://github.com/ryanwallace96/evosim',
    license='MIT License',
    packages=find_packages(),
    install_requires=['numpy'],
    ext_modules = cythonize('opinet/following_c.pyx'),
    include_dirs=[np.get_include()],
    extra_compile_args=["-w"],
    include_package_data=True,
    keywords=('evolution', 'simulation', 'simulator'),
    classifiers=[  
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
    ]
)