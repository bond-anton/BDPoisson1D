cdef class Function(object):
    cpdef double[:] evaluate(self, double[:] x)


cdef class ConstantFunction(Function):
    cdef:
        double __c


cdef class ZeroFunction(ConstantFunction):
    pass


cdef class InterpolateFunction(Function):
    cdef:
        double[:] __x, __y
        int __n


cdef class InterpolateFunctionMesh(InterpolateFunction):
    pass


cdef class Functional(Function):
    cdef:
        Function __f
        Function __result


cdef class PowFunction(Functional):
    cdef:
        double __pow


cdef class ScaledFunction(Functional):
    cdef:
        double __scale


cdef class NumericGradient(Functional):
    pass


cdef class BinaryFunctional(Functional):
    cdef:
        Function __p


cdef class FunctionSum(BinaryFunctional):
    pass


cdef class FunctionDifference(BinaryFunctional):
    pass


cdef class FunctionMultiplication(BinaryFunctional):
    pass


cdef class FunctionDivision(BinaryFunctional):
    pass
