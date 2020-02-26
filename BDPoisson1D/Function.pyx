from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone
from ._helpers cimport gradient1d


cdef class Function(object):

    cpdef double[:] evaluate(self, double[:] x):
        return x


cdef class InterpolateFunction(Function):

    def __init__(self, double[:] x, double[:] y):
        super(InterpolateFunction, self).__init__()
        self.__x = x
        self.__y = y
        self.__n = x.shape[0]

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def n(self):
        return self.__n

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int n = x.shape[0]
            int i, j = 1
            array[double] y, template = array('d')
        y = clone(template, n, zero=False)
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

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        return self.__f.evaluate(x)

    @property
    def f(self):
        return self.__f

    @f.setter
    def f(self, Function new_f):
        self.__f = new_f


cdef class NumericGradient(Functional):
    def __init__(self, Function f):
        super(NumericGradient, self).__init__(f)

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int n = x.shape[0]
            int i, j = 1
            array[double] y, template = array('d')
            double[:] dy
        if not isinstance(self.__f, InterpolateFunction):
            return gradient1d(self.__f.evaluate(x), x)
        else:
            y = clone(template, n, zero=False)
            dy = gradient1d(self.__f.y, self.__f.x)
            for i in range(n):
                while x[i] > self.__f.x[j] and j < self.__f.n - 1:
                    j += 1
                y[i] = dy[j-1] + (x[i] - self.__f.x[j-1]) * (dy[j] - dy[j-1]) / (self.__f.x[j] - self.__f.x[j-1])
            return y
