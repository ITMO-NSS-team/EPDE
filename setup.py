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
    """
    Reads the content of a file, ensuring proper handling of character encoding.
    
    This is essential for reliably loading equation definitions, data, or
    configuration files used within the EPDE framework.
    
    Args:
      *names: Path segments to the file, relative to the directory of the current file.
      **kwargs: Keyword arguments. May include 'encoding' to specify the file encoding.
    
    Returns:
      str: The content of the file as a string.
    """
    with open(
            join(dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


def extract_requirements(file_name):
    """
    Extracts and filters lines from a file, treating each line as a potential requirement.
    
    This function is used to prepare a list of requirements from a specified file,
    which is a preliminary step for the equation discovery process. By removing
    empty lines and comments, it ensures that only valid requirements are considered
    during the search for governing equations.
    
    Args:
        file_name (str): The name of the file containing the requirements.
    
    Returns:
        list[str]: A list of strings, where each string represents a valid requirement
                   extracted from the file. Empty lines and comments are excluded.
    """
    return [r for r in read(file_name).split('\n') if r and not r.startswith('#')]


def get_requirements():
    """
    Extracts and returns the requirements from a file.
    
    This function is crucial for setting up the necessary environment
    by identifying and listing the required Python packages.
    
    Args:
        None
    
    Returns:
        list: A list of strings, where each string is a requirement.
    """
    requirements = extract_requirements('requirements.txt')
    return requirements

setup(
      name = 'epde',
      version = '1.2.18',
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
      packages = find_packages(include = ['epde', 'epde.cache', 'epde.control', 
                                          'epde.interface', 'epde.integrate',
                                          'epde.optimizers', 'epde.optimizers.moeadd', 
                                          'epde.optimizers.single_criterion', 'epde.operators.common',
                                          'epde.operators', 'epde.operators.utils',
                                          'epde.operators.utils.parameters',
                                          'epde.operators.multiobjective', 
                                          'epde.operators.singleobjective', 'epde.preprocessing', 
                                          'epde.parametric', 'epde.structure', 
                                          'epde.solver', 'epde.solver.callbacks', 'epde.solver.optimizers']),
      include_package_data = True,                               
      python_requires =' >=3.8',
      )
