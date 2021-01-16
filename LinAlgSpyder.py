import numpy as np 
# 4x + 3y = 20 and -5x +9y = 26  convert to array and take inverse and do the dot product
m_list = [[4, 3], [-5, 9]]
A = np.array(m_list)
inverseArray = np.linalg.inv(A)
B = np.array([20,26])
result = inverseArray.dot(B)
print(result)

A = np.array([[4, 3, 2], [-2, 2, 3], [3, -5, 2]])
B = np.array([25, -10, -4])
result2 = np.linalg.solve(A,B)
print(result2)


from sympy import * 
  
M = Matrix([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]]) 
M_rref = M.rref()   
print("The Row echelon form of matrix M and the pivot columns : {}".format(M_rref)) 