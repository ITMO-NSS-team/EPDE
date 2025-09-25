```markdown
# Benchmarking

## Overview

This module focuses on applying the EPDE (Equation Parameter Discovery and Estimation) framework to identify governing differential equations for well-known physical systems, such as the Burgers', KdV, and Wave equations. It provides tools for configuring and executing EPDE searches, evaluating the discovered equations, and comparing the results against known solutions. The module supports both single-objective and multi-objective optimization modes.

## Purpose

The primary purpose of this module is to benchmark the EPDE framework's ability to rediscover known differential equations from data. It includes functionalities for:

- Running equation searches for the Burgers', KdV, and Wave equations using the EPDE algorithm.
- Configuring search parameters, preprocessors, and custom tokens for each equation.
- Evaluating the identified equations and reporting the optimization metric.
- Translating and evaluating partial differential equations using EPDE.
- Simulating and analyzing dynamical systems like the Lorenz system and Lotka-Volterra model.
- Visualizing solutions of these systems through 2D and 3D plots.
- Discovering governing equations from simulation data and writing Pareto front solutions to files.
```