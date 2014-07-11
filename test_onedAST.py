import math
from quadrature import gauss_legendre
from twoscalecoeffs import twoscalecoeffs,phi
from tensor import Vector, Matrix
import oned as OD
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
k = 5 # order of wavelet
thresh = 1e-12 # truncation threshold
f1 = ODT.FunctionAST(k,thresh,test1)    
f2 = ODT.FunctionAST(k,thresh,test2)
f4 = ODT.FunctionAST(k,thresh,sinn)

f1.compress()
f2.compress()

resultAST = ODT.FunctionAST(k,thresh)
newresult = ODT.FunctionAST(k,thresh)

n0 = ODT.Node(f1)
n1 = ODT.Node(f2)
n2 = ODT.Node(f2)
n4 = ODT.Node(f4)
n6 = ODT.Node(f4)

#Reconstruct ASTs n1,n2,n22,n3
AST0 = [2,n0]
AST1 = [2,n1]
AST2 = [2,n2]

#Differentiation of n6
AST5 = [3,n6]

#Create a full AST that computes (f1+f2)*(f2+f4+diff(f4))
AST_ADD0 = [0,AST0,AST2]
AST_ADD1 = [0,AST1,n4,AST5]
AST_MUL = [1,AST_ADD0,AST_ADD1]

#compute using onedAST
resultAST.traverse_tree(AST_MUL)

#Compute (f1+f2)*(f2+f4+diff(f4)) using original madpy without AST
r_add0 = f1 + f2
r_add1 = f2 + f4
r_diff = f4.diff()
r_add5 = r_add1 + r_diff
r_prod = r_add0*r_add5
r_prod.reconstruct()

df = resultAST - r_prod
print "Norm of Difference is ", df.norm2()

