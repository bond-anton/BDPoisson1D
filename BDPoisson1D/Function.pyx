import numpy as np

from cython cimport boundscheck, wraparound


cdef class Function(object):

    cpdef double[:] evaluate(self, double[:] x):
        return x


cdef class InterpolateFunction(Function):

    def __init__(self, double[:] x, double[:] y):
        super(InterpolateFunction, self).__init__()
        self.__x = x
        self.__y = y
        self.__n = x.size

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int n = x.size
            int i, j = 1
            double[:] y = np.zeros(n, dtype=np.double)
        for i in range(n):
            while x[i] > self.__x[j] and j < self.__n - 1:
                j += 1
            y[i] = self.__y[j-1] + (x[i] - self.__x[j-1]) * \
                   (self.__y[j] - self.__y[j-1]) / (self.__x[j] - self.__x[j-1])
        return y


cdef class Functional(Function):

    def __init__(self, Function f):
        super(Functional, self).__init__()
        self.__f = f

    @property
    def f(self):
        return self.__f

    @f.setter
    def f(self, Function new_f):
        self.__f = new_f


cdef class NumericDiff(Functional):

    def __init__(self, Function f):
        super(NumericDiff, self).__init__(f)

    cpdef double[:] evaluate(self, double[:] x):
        return np.gradient(self.__f.evaluate(x), x, edge_order=2)
