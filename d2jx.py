import functools
import operator
import collections
import itertools

#import numpy as np

import jax
import jax.numpy as np

import pylab as plt

import dn, ease

# Constants

ORIGIN = np.array((0, 0))

X = np.array((1, 0))
Y = np.array((0, 1))

UP = Y

# SDF Class

_ops = {}

class SDF2:
    def __init__(self, f):
        self.f = f
    def __call__(self, p):
        return self.f(p).reshape((-1, 1))
    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        raise AttributeError
    def __or__(self, other):
        return union(self, other)
    def __and__(self, other):
        return intersection(self, other)
    def __sub__(self, other):
        return difference(self, other)
    def k(self, k=None):
        self._k = k
        return self

def sdf2(f):
    def wrapper(*args, **kwargs):
        return SDF2(f(*args, **kwargs))
    return wrapper

def op2(f):
    def wrapper(*args, **kwargs):
        return SDF2(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

def op23(f):
    def wrapper(*args, **kwargs):
        return d3.SDF3(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

# Helpers

def _length(a):
    return np.linalg.norm(a, axis=1)

def _normalize(a):
    return a / np.linalg.norm(a)

def _dot(a, b):
    return np.sum(a * b, axis=1)

def _cross(a, b):
    return np.cross(a,b,axis=1)

def _vec(*arrs):
    return np.stack(arrs, axis=-1)

_min = np.minimum
_max = np.maximum

# Primitives

@sdf2
def circle(radius=1, center=ORIGIN):
    def f(p):
        return _length(p - center) - radius
    return f

@sdf2
def line(normal=UP, point=ORIGIN):
    normal = _normalize(normal)
    def f(p):
        return np.dot(point - p, normal)
    return f

@sdf2
def slab(x0=None, y0=None, x1=None, y1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(line(X, (x0, 0)))
    if x1 is not None:
        fs.append(line(-X, (x1, 0)))
    if y0 is not None:
        fs.append(line(Y, (0, y0)))
    if y1 is not None:
        fs.append(line(-Y, (0, y1)))
    return intersection(*fs, k=k)

@sdf2
def rectangle(size=1, center=ORIGIN, a=None, b=None):
    if a is not None and b is not None:
        a = np.array(a)
        b = np.array(b)
        size = b - a
        center = a + size / 2
        return rectangle(size, center)
    size = np.array(size)
    def f(p):
        q = np.abs(p - center) - size / 2
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0)
    return f

@sdf2
def rounded_rectangle(size, radius, center=ORIGIN):
    try:
        r0, r1, r2, r3 = radius
    except TypeError:
        r0 = r1 = r2 = r3 = radius
    def f(p):
        x = p[:,0]
        y = p[:,1]
        r = np.zeros(len(p)).reshape((-1, 1))
        r[np.logical_and(x > 0, y > 0)] = r0
        r[np.logical_and(x > 0, y <= 0)] = r1
        r[np.logical_and(x <= 0, y <= 0)] = r2
        r[np.logical_and(x <= 0, y > 0)] = r3
        q = np.abs(p) - size / 2 + r
        return (
            _min(_max(q[:,0], q[:,1]), 0).reshape((-1, 1)) +
            _length(_max(q, 0)).reshape((-1, 1)) - r)
    return f

@sdf2
def equilateral_triangle():
    def f(p):
        k = 3 ** 0.5
        p = _vec(
            np.abs(p[:,0]) - 1,
            p[:,1] + 1 / k)
        w = p[:,0] + k * p[:,1] > 0
        q = _vec(
            p[:,0] - k * p[:,1],
            -k * p[:,0] - p[:,1]) / 2
        p = np.where(w.reshape((-1, 1)), q, p)
        p = _vec(
            p[:,0] - np.clip(p[:,0], -2, 0),
            p[:,1])
        return -_length(p) * np.sign(p[:,1])
    return f

@sdf2
def hexagon(r):
    r *= 3 ** 0.5 / 2
    def f(p):
        k = np.array((3 ** 0.5 / -2, 0.5, np.tan(np.pi / 6)))
        p = np.abs(p)
        p -= 2 * k[:2] * _min(_dot(k[:2], p), 0).reshape((-1, 1))
        p -= _vec(
            np.clip(p[:,0], -k[2] * r, k[2] * r),
            np.zeros(len(p)) + r)
        return _length(p) * np.sign(p[:,1])
    return f

@sdf2
def rounded_x(w, r):
    def f(p):
        p = np.abs(p)
        q = (_min(p[:,0] + p[:,1], w) * 0.5).reshape((-1, 1))
        return _length(p - q) - r
    return f

@sdf2
def polygon(points):
    points = [np.array(p) for p in points]
    def f(p):
        n = len(points)
        d = _dot(p - points[0], p - points[0])
        s = np.ones(len(p))
        for i in range(n):
            j = (i + n - 1) % n
            vi = points[i]
            vj = points[j]
            e = vj - vi
            w = p - vi
            b = w - e * np.clip(np.dot(w, e) / np.dot(e, e), 0, 1).reshape((-1, 1))
            d = _min(d, _dot(b, b))
            c1 = p[:,1] >= vi[1]
            c2 = p[:,1] < vj[1]
            c3 = e[0] * w[:,1] > e[1] * w[:,0]
            c = _vec(c1, c2, c3)
            s = np.where(np.all(c, axis=1) | np.all(~c, axis=1), -s, s)
        return s * np.sqrt(d)
    return f

# Positioning

@op2
def translate(other, offset):
    def f(p):
        return other(p - offset)
    return f

@op2
def scale(other, factor):
    try:
        x, y = factor
    except TypeError:
        x = y = factor
    s = (x, y)
    m = min(x, y)
    def f(p):
        return other(p / s) * m
    return f

@op2
def rotate(other, angle):
    s = np.sin(angle)
    c = np.cos(angle)
    m = 1 - c
    matrix = np.array([
        [c, -s],
        [s, c],
    ]).T
    def f(p):
        return other(np.dot(p, matrix))
    return f

@op2
def circular_array(other, count):
    angles = [i / count * 2 * np.pi for i in range(count)]
    return union(*[other.rotate(a) for a in angles])

# Alterations

@op2
def elongate(other, size):
    def f(p):
        q = np.abs(p) - size
        x = q[:,0].reshape((-1, 1))
        y = q[:,1].reshape((-1, 1))
        w = _min(_max(x, y), 0)
        return other(_max(q, 0)) + w
    return f


# Common

union = op2(dn.union)
difference = op2(dn.difference)
intersection = op2(dn.intersection)
blend = op2(dn.blend)
negate = op2(dn.negate)
dilate = op2(dn.dilate)
erode = op2(dn.erode)
shell = op2(dn.shell)
repeat = op2(dn.repeat)


# view 2d sdfs

# def cartesian_product(*arrays):
#     la = len(arrays)
#     dtype = np.result_type(*arrays)
#     arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
#     for i, a in enumerate(np.ix_(*arrays)):
#         arr.at[...,i].set(a)
#     return arr.reshape(-1, la)

def _cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    return np.array([ai.ravel() for ai in np.meshgrid(*arrays)]).T



def _estimate_bounds(sdf):
    # TODO: raise exception if bound estimation fails
    s = 16
    x0 = y0 = -1e9
    x1 = y1 = 1e9
    prev = None
    for i in range(32):
        X = np.linspace(x0, x1, s)
        Y = np.linspace(y0, y1, s)
        d = np.array([X[1] - X[0], Y[1] - Y[0]])
        threshold = np.linalg.norm(d) / 2
        if threshold == prev:
            break
        prev = threshold
        P = _cartesian_product(X, Y)
        volume = sdf(P).reshape((len(X), len(Y)))
        where = np.argwhere(np.abs(volume) <= threshold)
        x1, y1 = np.array((x0, y0)) + where.max(axis=0) * d + d / 2
        x0, y0 = np.array((x0, y0)) + where.min(axis=0) * d - d / 2
    return ((x0, y0), (x1, y1))


def grid_2d(
    sdf, w=500, h=500,
    bounds=None):

    if bounds is None:
        bounds= _estimate_bounds(sdf)
    (x0, y0), (x1, y1) = bounds

    X = np.linspace(x0, x1, w)
    Y = np.linspace(y0, y1, h)
    extent = (X[0], X[-1], Y[0], Y[-1])
    P = _cartesian_product(X, Y)

    return sdf(P).reshape((w,h)), extent

def show_2d(*args, **kwargs):
    import matplotlib.pyplot as plt
    show_abs = kwargs.pop('abs', False)
    a, extent = grid_2d(*args, **kwargs)
    if show_abs:
        a = np.abs(a)
    im = plt.imshow(a, extent=extent, origin='lower')
    #plt.xlabel(axes[0])
    #plt.ylabel(axes[1])
    plt.colorbar(im)
    plt.show()

