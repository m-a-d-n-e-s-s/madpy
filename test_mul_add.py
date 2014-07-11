#Tests for (f1+f2)*f4 using onedAST
import math
from quadrature import gauss_legendre
from twoscalecoeffs import twoscalecoeffs,phi
from tensor import Vector, Matrix
from oned import Function
import onedAST as ODT

def sinn(x):
    return math.sin(2.0*3.1416*x)

def test1(x):
    ''' gaussian with square normalized to 1 '''
    a = 500.0
    return pow(2*a/math.pi,0.25)*math.exp(-a*(x-0.5)**2)

def test2(x):
    ''' superposition of multiple gaussians '''
    return test1(x-0.3) + test1(x)  + test1(x+0.3)

npt = 20 # No. points to sample on test printing
k = 10 # order of wavelet
thresh = 1e-12 # truncation threshold

f1 = ODT.FunctionAST(k,thresh,test1)    
f2 = ODT.FunctionAST(k,thresh,test2)
f4 = ODT.FunctionAST(k,thresh,sinn)

f1.reconstruct()
f2.reconstruct()
f4.reconstruct()

resultAST = ODT.FunctionAST(k,thresh)

n1 = ODT.Node(f1)
n2 = ODT.Node(f2)
n4 = ODT.Node(f4)

AST0 = [0,n1,n2]
AST=[1,AST0,n4]
resultAST.traverse_tree(AST)

t1 = f1 + f2
t2 = f4
result = t1*t2

df = result- resultAST
print "The Norm of Difference is ", df.norm2()



