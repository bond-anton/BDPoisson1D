cdef class Function(object):
    cpdef double[:] evaluate(self, double[:] x)

cdef class NumericDiff(Function):
    cdef:
        Function __f

cdef class InterpolateFunction(Function):
    cdef:
        double[:] __x, __y
        int __n
