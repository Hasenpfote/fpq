#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup


def _load_requirements_from_file(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

def _install_requires():
    return _load_requirements_from_file('requirements.txt')

def _tests_require():
    return _load_requirements_from_file('requirements-test.txt')

def _doc_require():
    return _load_requirements_from_file('requirements-doc.txt')

def _long_description():
    with open('README.rst', 'r') as f:
        return f.read()

if __name__ == '__main__':
    setup(
        name='fpq',
        version='0.4.0',
        description='',
        long_description=_long_description(),
        author='Hasenpfote',
        author_email='Hasenpfote36@gmail.com',
        url='',
        download_url='',
        packages = ['fpq'],
        keywords=['',],
        classifiers=[
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Development Status :: 3 - Alpha',
            'Environment :: Other Environment',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        install_requires=_install_requires(),
        tests_require=_tests_require(),
        #test_suite='nose.collector',
        extras_require = {
            'test': _tests_require(),
            'doc': _doc_require(),
        },
    )