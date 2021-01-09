#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from setuptools import setup
import sys


# Get version without importing, which avoids dependency issues
def get_version():
    with open('fpq/version.py') as version_file:
        return re.search(r"""__version__\s+=\s+(['"])(?P<version>.+?)\1""",
                         version_file.read()).group('version')


def _long_description():
    with open('README.rst', 'r') as f:
        return f.read()


required=[
    'numpy>=1.11.2',
]

if sys.version_info >= (3, 5):
    required.extend([
        'numba>=0.38.0, <0.44.0',
        'llvmlite<0.29.0'
    ])


if __name__ == '__main__':
    setup(
        name='fpq',
        version=get_version(),
        description='This package provides modules for manipulating floating point numbers quantization using NumPy.',
        long_description=_long_description(),
        author='Hasenpfote',
        author_email='Hasenpfote36@gmail.com',
        url='https://github.com/Hasenpfote/fpq',
        download_url='',
        packages = ['fpq'],
        keywords=['quantization', 'floating-point', 'vector', 'quaternion'],
        classifiers=[
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Development Status :: 4 - Beta',
            'Environment :: Other Environment',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        python_requires='>=3.4',
        install_requires=required,
    )
