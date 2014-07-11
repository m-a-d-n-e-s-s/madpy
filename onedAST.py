import math
from quadrature import gauss_legendre
from twoscalecoeffs import twoscalecoeffs,phi
from tensor import Vector, Matrix
from oned import Function

DEBUG=0
DEBUG2=0
DEBUGD=0
DEBUGR=0

ERROR_CODE = -1
ADD_OPERATOR = 0
MULTIPLY_OPERATOR = 1
RECONSTRUCT_OPERATOR = 2
DIFF_OPERATOR = 3

#This class inherits Function class. It implements a single pass
#AST evaluation algorithm. The algorithm can evaluate an
#expression composed of addition, multiplication, diffrentiation
#in a single tree traversal from top to bottom of the output
#function tree. The expression is represented using an AST, and
#associative reodering is allowed for addition, multiplication and
#subtraction. During the tree traversal, the AST is evaluated and
#if possible simplified, before passing down to the children
#nodes.

class Node:
    #initialize node 
    def __init__(self, 
                 function_name, 
                 is_inter=0, 
                 has_coefficient=0, coefficient=0, 
                 isready=0, 
                 n=0, l=0):
        self.is_intermediate = is_inter
        self.function = function_name
        self.has_coefficient  = has_coefficient
        self.is_ready = isready
        self.coeff = coefficient
        self.level = n
        self.translation = l
    
    #return a deep copy of the node
    def copy(self):
        new_Node = Node(self.function, 
                        self.is_intermediate, 
                        self.has_coefficient, self.coeff, 
                        self.is_ready, 
                        self.level, self.translation)
        return new_Node
    
    
class FunctionAST(Function):

    def __init__(self,k,thresh,f=None,initial_level=2):
        Function.__init__(self,k,thresh,f,initial_level=2)

    def create_node(self,coeff,n,l,is_ready=1):
        return Node(None, 1, 1, coeff, is_ready, n, l)

    #for co-ordinates at box nn,ll, evaluates the value using 
    #coefficients at box n,l
    def evaluate_at_box(self,k,coeff,n,l,nn,ll,x):
        if isinstance(x,list) or isinstance(x,Vector):
            value_list = []
            for i in range(len(x)):
                coordinate = (x[i]+ll)*(2.0**(n-nn))-l
                p = Vector(phi(coordinate,k))                                                                                                  
                value_list.append(coeff.inner(p)*math.sqrt(2.0**n))
            return Vector(value_list)
        else:
            coordinate = (x + ll)*(2.0**(n-nn))-l
            p = Vector(phi(coordinate,self.k))                                                                                                  
            value = coeff.inner(p)*math.sqrt(2.0**n)
            return value
            
    def evaluate_function(self,x,n,l):
        coordinate = (x[i]+l)*(2.0**(-n))
        return self.f(coordinate)

    #for values at the quadrature points at box n,l returns coefficients at
    #the same box
    def quad_values_to_coeff(self,values,n,l,transform_matrix):
        coeff = (values*transform_matrix).scale(math.sqrt(2**(-n)))
        return coeff
    
    #given a parent node with coefficients, returns coefficients at the child box nn,ll
    def get_coeff_from_parent1(self,coeff,n,l,nn,ll):
        parent_coeff = coeff

        if not self.is_parent(n,l,nn,ll):
            return -1
        else:
            computed_values = self.evaluate_at_box(self.k,parent_coeff,n,l,nn,ll,self.quad_x)
            computed_coeff = self.quad_values_to_coeff(computed_values,nn,ll,self.quad_phiw)
            return computed_coeff
    

    #given a parent node with coefficients, returns coefficients at the child box nn,ll
    def get_coeff_from_parent(self,nd,nn,ll):
        parent_coeff = nd.coeff
        n = nd.level
        l = nd.translation

        if not self.is_parent(n,l,nn,ll):
            return -1
        else:
            computed_values = self.evaluate_at_box(self.k,parent_coeff,n,l,nn,ll,self.quad_x)
            computed_coeff = self.quad_values_to_coeff(computed_values,nn,ll,self.quad_phiw)
            return computed_coeff

    def multiply_coefficients(self, coeff1, coeff2, n):

        #coeff_values stores values of the function at boxed
        #defined by coeff
        coeff1_values = coeff1 * self.quad_phiT
        coeff2_values = coeff2 * self.quad_phiT
        
        #do the multiplication on the values for each of the boxes
        coeff1_values.emul(coeff2_values)

        # scale factor for this level = sqrt((2^d)^(n+1))
        scale_factor = math.sqrt(2.0**n)
                
        #convert values back to coefficients for each of the boxes
        result_coeff = (coeff1_values * self.quad_phiw).scale(scale_factor)

        return result_coeff

    def add_coefficients(self,coeff1,coeff2):
        temp = Vector(coeff1)
        temp.gaxpy(1.0, coeff2, 1.0)
        return temp
    
    def add_ready_operands(self, evaluated_operands_list, ready_operand_indices, n, l):

        le = len(evaluated_operands_list)
        lr = len(ready_operand_indices)
        
        result_coeff = self.get_coeff_from_parent(evaluated_operands_list[ready_operand_indices[0]],n,l)

        #iterating over operands whose coefficients are available
        for j in range (len(ready_operand_indices)-1):
            coeff = self.get_coeff_from_parent(evaluated_operands_list[ready_operand_indices[j+1]],n,l)
            
            result_coeff = self.add_coefficients(result_coeff,coeff)

        #sorts in reverse order so that operands can be popped off
        #of the evaluated_operands_list without messing up the
        #indices
        for i1  in range(lr):
            for i2 in range(lr-1):
                temp1 = ready_operand_indices[i2]
                temp2 = ready_operand_indices[i2+1]
                if temp1 < temp2:
                    ready_operand_indices[i2] = temp2
                    ready_operand_indices[i2+1] = temp1                    
        
        for j in range (len(ready_operand_indices)):
            evaluated_operands_list.pop(ready_operand_indices[j])
            
        #not implemeted yet
        result_node=self.create_node(result_coeff,n,l)

            #if all the operands are available for computing
            #then the operation is complete. Just replace the
            #operation node with the result node
        if le == lr:
            if n == self.max_level:
                result_node.is_ready = 1
            return result_node
            
        #the operation is not complete yet. Some operands
        #are still at internal nodes`
        else:
            evaluated_operands_list.insert(0,ADD_OPERATOR)
            evaluated_operands_list.append(result_node)
            return evaluated_operands_list

    def multiply_ready_operands(self, evaluated_operands_list, ready_operand_indices, n,l):
        #not implemented yet
        
        #FIXME create a null vector
        result_coeff = None
        result_node_list=[]
        #iterating over operands whose coefficients are available
        lr = len(ready_operand_indices)        

        for j in range (0,lr-1,2):
            #print lr, "In loop [",n,",",l,"]"
            coeff1 = self.get_coeff_from_parent(evaluated_operands_list[ready_operand_indices[j]],n,l)
            coeff2 = self.get_coeff_from_parent(evaluated_operands_list[ready_operand_indices[j+1]],n,l)
            result_coeff = self.multiply_coefficients(coeff1, coeff2, n)            

            #not implemeted yet
            result_node = self.create_node(result_coeff,n,l, 0)
            result_node_list.append(result_node)

        if lr % 2:
            #print "in modulo"
            result_node_list.append(evaluated_operands_list[ready_operand_indices[lr-1]])

        #sorts in reverse order so that operands can be popped off
        #of the evaluated_operands_list without messing up the
        #indices
        for i1  in range(lr):
            for i2 in range(lr-1):
                temp1 = ready_operand_indices[i2]
                temp2 = ready_operand_indices[i2+1]
                if temp1 < temp2:
                    ready_operand_indices[i2] = temp2
                    ready_operand_indices[i2+1] = temp1                    


        for j in range (len(ready_operand_indices)):
            evaluated_operands_list.pop(ready_operand_indices[j])

        #print result_node_list, len(result_node_list)
        evaluated_operands_list.extend(result_node_list)

        #if all the operands are available for computing
        #then the operation is complete. Just replace the
        #operation node with the result node
        if len(evaluated_operands_list) == 1:
            #print evaluated_operands_list[0]            

            if n==self.max_level:
                evaluated_operands_list[0].is_ready = 1

            return evaluated_operands_list[0]

        else:
            evaluated_operands_list.insert(0,MULTIPLY_OPERATOR)
            #print evaluated_operands_list
            return evaluated_operands_list

    def get_scaling_coeff(self,scaling_coeff, wavelet_coeff, n, l):
        k = self.k
        d = Vector(2*k)
        d[:k],d[k:] = scaling_coeff, wavelet_coeff

        is_odd = l - 2*(l/2)

        # apply the two scale relationship to get difference coeff
        # in 1d this is O(k^2) flops (in 3d this is O(k^4) flops)
        if not is_odd:
            s = d * self.hg0
        else:
            s = d * self.hg1
        return s

    def reconstruct_operation(self, evaluated_operand, n, l):
        
        if n == 0:
            return [RECONSTRUCT_OPERATOR, evaluated_operand]

        result_coeff = self.get_scaling_coeff(evaluated_operand.coeff, evaluated_operand.function.d[n-1][l/2],n,l)
        evaluated_operand.coeff = result_coeff
        evaluated_operand.level, evaluated_operand.translation = n,l
        
        if evaluated_operand.function.d[n].has_key(l):                
            temp = []
            temp.append(RECONSTRUCT_OPERATOR)
            temp.append(evaluated_operand)
            return temp
        else:
            return evaluated_operand

    def is_parent(self,parent_n, parent_l, child_n, child_l):

        difference_level = child_n - parent_n
        parent_of_child_l = child_l/(2**difference_level)

        if parent_of_child_l == parent_l:
            return 1
        else:
            return 0
        
    def diff_coefficients(self,left,center,right,n):
        r = self.rp*left + self.r0*center + self.rm*right
        return r.scale(2.0**n)

            
    def diff_operation(self, evaluated_operand, n, l):
        parent_n = evaluated_operand.level
        parent_l = evaluated_operand.translation
        center = self.get_coeff_from_parent(evaluated_operand,n,l)
        if 0:
            tright = evaluated_operand.function.get_coeffs(n,l)
            sum = 0.0
            for i in range(self.k):
                sum += (center.a[i]-tright.a[i])**2
                if sum > 10.0e-6:
                    print sum
        
        #if the evaluated operand has coefficients for a parent node of left
        if self.is_parent(parent_n,parent_l,n,l-1):
            left = self.get_coeff_from_parent(evaluated_operand,n,l-1)
            #find left coefficients from the function that is being differentiated
            if 0.0:
                tright = evaluated_operand.function.get_coeffs(n,l-1)
                sum = 0.0
                for i in range(self.k):
                    sum += (left.a[i]-tright.a[i])**2
                if sum > 0.0:
                    print sum

        else:
            left = evaluated_operand.function.get_coeffs(n,l-1)

        if self.is_parent(parent_n,parent_l,n,l+1):
            right = self.get_coeff_from_parent(evaluated_operand,n,l+1)
            if 0:
                tright = evaluated_operand.function.get_coeffs(n,l+1)
                sum = 0.0
                for i in range(self.k):
                    sum += (right.a[i]-tright.a[i])**2
                if sum > 0.0:
                    print sum
        else:
            right = evaluated_operand.function.get_coeffs(n,l+1)

        if center and left and right:
            coeff = self.diff_coefficients(left,center,right,n)

            if n == self.max_level:
                return self.create_node(coeff, n, l, 1)
            
            return self.create_node(coeff, n, l)

        else:
            temp =[]
            temp.append(DIFF_OPERATOR)
            temp.append(evaluated_operand)
            return temp


    def binary_operation(self,op, evaluated_operands_list, ready_operands_indices, n, l):

        if op == ADD_OPERATOR:
            return self.add_ready_operands(evaluated_operands_list, ready_operands_indices, n, l)

        elif op == MULTIPLY_OPERATOR:
            return self.multiply_ready_operands(evaluated_operands_list, ready_operands_indices, n, l)

        else:
            if DEBUG: 
                print "Unknown Operator",op

    def unary_operation(self, op, evaluated_operand, n, l):

        if op == RECONSTRUCT_OPERATOR:
            return self.reconstruct_operation(evaluated_operand, n, l)

        elif op == DIFF_OPERATOR:
            return self.diff_operation(evaluated_operand,n,l)

        else:
            if DEBUG: 
                print "Unknown Operator", op    

    def pre_compute(self, op, operand,n,l):

        is_ready = 0

        if not operand.has_coefficient:

            if DEBUG: 
                print "No Coefficient at [",n,",",l,"]"

            if operand.function.s[n].has_key(l):
                operand.coeff = operand.function.s[n][l]
                operand.level = n
                operand.translation = l

                if n == self.max_level:
                    is_ready = 1

                if DEBUG: 
                    print "[",n,",",l,"] Found Coefficient" , operand.coeff

                operand.has_coefficient = 1

        elif n > operand.level or n == self.max_level:
            is_ready = 1

        else:
            print "Dumb! Not supposed to happen, your coefficients are more refined that you are"
        
        if operand.has_coefficient:

            if op == ADD_OPERATOR or op == DIFF_OPERATOR or op == RECONSTRUCT_OPERATOR:
                operand.is_ready = 1

            if op == MULTIPLY_OPERATOR and is_ready == 1:
                operand.is_ready = 1
            
                

        return operand
    
    def evaluate_AST(self, AST, n=0, l=0):
        result_coeff = None
        evaluated_operands_list = []
        ready_operands_indices = [] 
        coeff = []
        
        #binary associative operator
        if len(AST) > 2:
            op = AST[0]
            num_operand = len(AST) - 1            

            for i in range (1,1+num_operand):
                operand = AST[i]
                
                #if the operand is an expression then evaluate that expression
                if not isinstance(operand,Node):
                    evaluated_operand = self.evaluate_AST(operand,n,l)

                #check if the node has coefficents ready at this level
                else:
                    evaluated_operand = self.pre_compute(op,operand,n,l)

                #add the evaluated operand to computed list
                evaluated_operands_list.append(evaluated_operand)

                #if the operand is a single function then check if
                #it has coefficients and add to the coeff list if
                #it does
                if isinstance(evaluated_operand,Node) and evaluated_operand.is_ready:
                    ready_operands_indices.append(i-1)

            #compute the sum for set of operands whose coefficients are available
            if(len(ready_operands_indices) > 1):
                if DEBUG: 
                    print "Computing Binary Operation"
                return self.binary_operation(op, evaluated_operands_list, ready_operands_indices, n, l)
            else:
                evaluated_operands_list.insert(0,op)
            
            return evaluated_operands_list
            
        #unary operation
        if(len(AST) == 2):
            op = AST[0]
            operand= AST[1]
            evaluated_operands_list = []

            #if the operand is an expression then evaluate that expression
            if not isinstance(operand,Node):
                evaluated_operand = self.evaluate_AST(operand,n,l)
                            
                #if the operand is a single function then just
                #evaluate it to see if it has coefficients, update
                #the node by assigning coefficient if it has any
            else:
                evaluated_operand = self.pre_compute(op, operand,n,l)
                if DEBUG: 
                    print evaluated_operand
            
            if isinstance(evaluated_operand,Node) and evaluated_operand.is_ready:
                return self.unary_operation(op, evaluated_operand, n, l)
            else:
                evaluated_operands_list.append(op)
                evaluated_operands_list.append(evaluated_operand)
                return evaluated_operands_list

    #makes a deep copy of the AST to avoid aliasing problems
    def copy_AST(self,AST):

        if not isinstance(AST,Node):
            length = len(AST)
            AST_copy = [AST[0]]

            for i in range(length-1):
                temp = self.copy_AST(AST[i+1])
                AST_copy.append(temp)
            return AST_copy

        else:
            return AST.copy()
     

    def traverse_tree(self, AST, n=0, l=0):
        #print AST[1].function.s[3][0]
        new_AST = self.evaluate_AST(AST, n,l)

        if isinstance(new_AST, Node) and new_AST.has_coefficient:
            if DEBUG: 
                print "Yay Computed result at [",n,",",l,"]"

            self.s[n][l] = new_AST.coeff

        elif not isinstance(new_AST,Node):
            if DEBUG: 
                print "Refining ..."
                print new_AST

            if DEBUG2:
                print "Refining at",n+1
                print new_AST

            AST_copy = self.copy_AST(new_AST)
            
            self.traverse_tree(new_AST,n+1,2*l)

            self.traverse_tree(AST_copy,n+1,2*l+1)
        else:
            if DEBUG: 
                print "evaluate_AST returned a node without coefficients"



    


        
