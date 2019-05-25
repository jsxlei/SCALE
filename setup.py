#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Thu 18 Oct 2018 02:42:37 PM CST

# File Name: setup.py
# Description:

"""

from setuptools import setup, find_packages
# import sys

# if sys.version_info[:2] < (3,6):
    # raise RuntimeError("Python version >=3.6 required.")
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='SCALE',
      version='1.0.1',
      description='Single-Cell ATAC-seq Analysis via Latent feature Extraciton',
      packages=find_packages(),

      author='Xiong Lei',
      author_email='jsxlei@gmail.com',
      url='https://github.com/jsxlei/SCALE',
      scripts=['SCALE.py'],
      install_requires=requirements,

      classifies=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
     ],
     )
