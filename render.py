import itertools
import functools
import collections
import operator

import jax
import jax.numpy as np

import pylab as plt

def camera_ray_direction(uv, campos, camtarget):
    pass


def length(p):
    return np.linalg.norm(p)

def sdsphere(p, r):
    return length(p) - r

def sdf(p):
    t = sdsphere(p-np.array([[0,0,10]]), 3.0)
    return t

def castray(ray_origin, ray_dir):
    t = 0
    while res := sdf(ray_origin + ray_dir*t) > tol*t:
        t = res

def castray_np(ray_origin, ray_dir):
    t = 0
    while res := sdf(ray_origin + ray_dir*t):

