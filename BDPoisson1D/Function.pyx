import numpy as np

from cython cimport boundscheck, wraparound


cdef class Function(object):

    cpdef double[:] evaluate(self, double[:] x):
        return x


cdef class NumericDiff(Function):

    def __init__(self, Function f):
        self.__f = f

    cpdef double[:] evaluate(self, double[:] x):
        return np.gradient(self.__f.evaluate(x), x, edge_order=2)


cdef class InterpolateFunction(Function):

    def __init__(self, double[:] x, double[:] y):
        self.__x = x
        self.__y = y
        self.__n = x.shape[0]
        super(InterpolateFunction, self).__init__()

    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int n = x.shape[0]
            int i, j = 1
            double[:] y = np.zeros(n, dtype=np.double)
        with boundscheck(False), wraparound(False):
            for i in range(n):
                while x[i] > self.__x[j] and j < self.__n - 1:
                    j += 1
                y[i] = y[j-1] + (x[i] - self.__x[j-1]) * (self.__y[j] - self.__y[j-1]) / (self.__x[j] - self.__x[j-1])
        return y
