cdef class Function(object):
    cpdef double[:] evaluate(self, double[:] x)

cdef class InterpolateFunction(Function):
    cdef:
        double[:] __x, __y
        int __n

cdef class Functional(Function):
    cdef:
        Function __f

cdef class NumericDiff(Functional):
    pass