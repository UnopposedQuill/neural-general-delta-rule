import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
 
def sigmoid_derivada(x):
    return x * (1.0 - x)
 
def tangente(x):
    return np.tanh(x)
 
def derivada_tangente(x):
    return 1.0 - x**2