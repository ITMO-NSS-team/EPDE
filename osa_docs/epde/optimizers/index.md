# Optimizers

## Overview

The Optimizers module provides a framework for constructing and applying evolutionary strategies to optimize equation discovery within the EPDE project. It includes tools for building customizable sequences of evolutionary operators, managing populations of candidate solutions, and implementing both single-criterion and multi-objective optimization algorithms. The module facilitates the creation of complex evolutionary search strategies through a modular block-based approach.

## Purpose

The primary purpose of the Optimizers module is to enable the automated discovery of governing differential equations from data by evolving equation structures that best fit the observed data. It offers functionalities for:

*   Constructing and linking evolutionary operators into flexible, modular strategies.
*   Managing populations of candidate solutions, including systems of equations.
*   Implementing single-criterion optimization using evolutionary algorithms, guided by a single objective function.
*   Performing multi-objective optimization using a Decomposition-based Evolutionary Algorithm (MOEA/D) to find Pareto-optimal solutions.
*   Defining custom stopping conditions for the evolutionary algorithm.
