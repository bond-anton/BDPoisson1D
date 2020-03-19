from libc.math cimport pow

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone
from ._helpers cimport gradient1d

from BDMesh.Mesh1D cimport Mesh1D
from BDMesh.TreeMesh1D cimport TreeMesh1D


cdef class Function(object):

    cpdef double[:] evaluate(self, double[:] x):
        return x


cdef class ConstantFunction(Function):

    def __init__(self, double c):
        super(ConstantFunction, self).__init__()
        self.__c = c

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, double c):
        self.__c = c

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = self.__c
        return y


cdef class ZeroFunction(ConstantFunction):

    def __init__(self):
        super(ZeroFunction, self).__init__(0.0)

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        return clone(array('d'), x.shape[0], zero=True)


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
            int i, j = 1, n = x.shape[0]
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            while x[i] > self.__x[j] and j < self.__n - 1:
                j += 1
            y[i] = self.__y[j-1] + (x[i] - self.__x[j-1]) * \
                   (self.__y[j] - self.__y[j-1]) / (self.__x[j] - self.__x[j-1])
        return y


cdef class InterpolateFunctionMesh(InterpolateFunction):

    def __init__(self, mesh):
        if isinstance(mesh, TreeMesh1D):
            flat_mesh = mesh.flatten()
            x = flat_mesh.physical_nodes()
            y = flat_mesh.solution
        elif isinstance(mesh, Mesh1D):
            x = mesh.physical_nodes()
            y = mesh.solution
        super(InterpolateFunctionMesh, self).__init__(x, y)


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
    def f(self, Function f):
        self.__f = f


cdef class ScaledFunction(Functional):

    def __init__(self, Function f, double scale):
        super(ScaledFunction, self).__init__(f)
        self.__scale = scale

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, double scale):
        self.__scale = scale

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = self.__scale * f_y[i]
        return y


cdef class PowFunction(Functional):

    def __init__(self, Function f, double exp):
        super(PowFunction, self).__init__(f)
        self.__exp = exp

    @property
    def exp(self):
        return self.__exp

    @exp.setter
    def exp(self, double exp):
        self.__exp = exp

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = pow(f_y[i], self.__exp)
        return y


cdef class NumericGradient(Functional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, j = 1, n = x.shape[0]
            array[double] y
            double[:] dy
        if not isinstance(self.__f, InterpolateFunction):
            return gradient1d(self.__f.evaluate(x), x)
        else:
            y = clone(array('d'), n, zero=False)
            dy = gradient1d(self.__f.y, self.__f.x)
            for i in range(n):
                while x[i] > self.__f.x[j] and j < self.__f.n - 1:
                    j += 1
                y[i] = dy[j-1] + (x[i] - self.__f.x[j-1]) * (dy[j] - dy[j-1]) / (self.__f.x[j] - self.__f.x[j-1])
            return y


cdef class BinaryFunctional(Functional):

    def __init__(self, Function f, Function p):
        super(BinaryFunctional, self).__init__(f)
        self.__p = p

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, Function p):
        self.__p = p


cdef class FunctionSum(BinaryFunctional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            double[:] p_y = self.__p.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = f_y[i] + p_y[i]
        return y


cdef class FunctionDifference(BinaryFunctional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            double[:] p_y = self.__p.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = f_y[i] - p_y[i]
        return y


cdef class FunctionMultiplication(BinaryFunctional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            double[:] p_y = self.__p.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = f_y[i] * p_y[i]
        return y


cdef class FunctionDivision(BinaryFunctional):

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:] evaluate(self, double[:] x):
        cdef:
            int i, n = x.shape[0]
            double[:] f_y = self.__f.evaluate(x)
            double[:] p_y = self.__p.evaluate(x)
            array[double] y = clone(array('d'), n, zero=False)
        for i in range(n):
            y[i] = f_y[i] / p_y[i]
        return y
