# Single Criterion Optimization Module

## Overview

This module focuses on evolutionary strategies for single-criterion optimization within the EPDE framework. It provides tools for constructing and managing populations of candidate solutions, defining stopping conditions for the evolutionary algorithm, and implementing the core optimization loop. The module includes classes for defining evolutionary strategies, managing populations, and constructing populations of systems of equations. It also offers a baseline evolutionary strategy director for assembling a basic evolutionary algorithm pipeline.

## Purpose

The primary purpose of this module is to provide a set of tools and classes for performing single-criterion optimization using evolutionary algorithms within the EPDE framework. This involves evolving populations of candidate solutions to find the best fit to the observed data, guided by a single objective function. The module supports defining custom stopping conditions, constructing populations of systems of equations with specific properties, and implementing the core optimization loop. It aims to automate the process of identifying governing differential equations from data by evolving equation structures that minimize a single error metric.
