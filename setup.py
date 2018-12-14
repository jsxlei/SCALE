#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Thu 18 Oct 2018 02:42:37 PM CST

# File Name: setup.py
# Description:

"""

from distutils.core import setup
import sys

if sys.version_info[:2] < (3,6):
    raise RuntimeError("Python version >=3.6 required.")

setup(name='SCALE',
      version='0.9',
      description='Single-Cell ATAC-seq Analysis via Latent feature Extraciton',
      author='Xiong Lei',
      author_email='jsxlei@gmail.com',
      url='https://github.com/jsxlei/SCALE/',
      packages=['scale', ],
      scripts=['SCALE.py'],
     )
