#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:20:30 2019

@author: cassio
"""

import numpy as np


def sigmoid_calculate(z):
    sigmoide = 1.0 / (1.0 + np.exp(-z))
    return sigmoide