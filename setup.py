#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:36:47 2021

@author: mike_ubuntu
"""

from setuptools import setup, find_packages
import os
from pathlib import Path
import pathlib

HERE = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
SHORT_DESCRIPTION = 'Framework for automated ordinary/partial differential equation discovery'
LONG_DESCRIPTION = (Path(os.path.join(HERE, "README.rst"))).read_text()


def read(*names, **kwargs):
    with open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


def extract_requirements(file_name):
    return [r for r in read(file_name).split('\n') if r and not r.startswith('#')]


def get_requirements():
    requirements = extract_requirements('requirements.txt')
    return requirements

setup(
      name = 'epde',
      version = '1.1.20',
      author = 'Mikhail Maslyaev',
      author_email = 'miklemas@list.ru',
      description=SHORT_DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/x-rst",
      classifiers = [      
              'Development Status :: 3 - Alpha',
              'Programming Language :: Python :: 3',
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent',
      ],
      packages = find_packages(include = ['epde', 'epde.cache', 'epde.interface', 'epde.moeadd', 
                                          'epde.operators', 'epde.prep']), 
      python_requires =' >=3.6'
      )
