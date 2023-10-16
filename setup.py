#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:36:47 2021

@author: mike_ubuntu
"""

from setuptools import setup, find_packages
from os.path import dirname, join
from pathlib import Path
import pathlib

HERE = pathlib.Path(__file__).parent.resolve()
README = Path(HERE, 'README.rst').read_text(encoding='utf-8')
SHORT_DESCRIPTION = 'Data-driven dynamical system and differential equations discovery framework'

# Get the long description from the README file


def read(*names, **kwargs):
    with open(
            join(dirname(__file__), *names),
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
      version = '1.2.12',
      description = SHORT_DESCRIPTION,
      long_description="PLACEHOLDER",
#      long_description_content_type='text/x-rst',
      author = 'Mikhail Maslyaev',
      author_email = 'miklemas@list.ru',
      classifiers = [      
              'Development Status :: 3 - Alpha',
              'Programming Language :: Python :: 3',
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent',
      ],
      packages = find_packages(include = ['epde', 'epde.cache', 'epde.interface', 
                                          'epde.optimizers', 'epde.optimizers.moeadd', 
                                          'epde.optimizers.single_criterion', 'epde.operators.common',
                                          'epde.operators', 'epde.operators.utils',
                                          'epde.operators.utils.parameters',
                                          'epde.operators.multiobjective', 
                                          'epde.operators.singleobjective', 'epde.preprocessing', 
                                          'epde.parametric', 'epde.structure', 'epde.solver']),
      include_package_data = True,                               
      python_requires =' >=3.8',
      )
