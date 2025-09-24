# KdV Module

## Overview

The KdV module focuses on discovering the Korteweg-de Vries (KdV) equation from data using the EPDE (Equation Discovery) framework. It provides functionalities for loading and preparing KdV equation data from various sources (CSV, MAT files), adding noise to the data, and setting up EPDE searches with custom tokens and preprocessors. The module also includes tools for comparing symbolic equations, preparing sub-operators for compound fitness operators, and testing the performance of EPDE searches and symbolic genetic algorithms (SGA) on the KdV equation. It supports using pre-trained Physics-Informed Neural Networks (PINNs) and SINDy (Sparse Identification of Nonlinear Dynamics) for equation discovery.

## Purpose

The primary purpose of the KdV module is to facilitate the automated discovery of the KdV equation from noisy data. It offers a suite of tools for data handling, noise injection, equation search setup, and performance evaluation, enabling users to identify the KdV equation using different EPDE-based approaches, including finite difference preprocessors, symbolic genetic algorithms, and SINDy. The module also provides functionalities for comparing discovered equations with known correct and incorrect equations, ensuring the accuracy and reliability of the equation discovery process.
