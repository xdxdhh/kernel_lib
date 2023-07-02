import numpy as np
import kernel



class Kernel_matrix(object):
    def __init__(self, X, kernel_type, params="auto") -> None: #X is input data, kernel  is string, params is a list of params or "auto"
        """ initialization of kernel matrix and size parameter n and params"""
        self.n = np.shape(X)[0]
        self.shape = [self.n, self.n]
        self.K = np.zeros((self.n, self.n)) #allocate matrix
        #switch podle kernel functions: "gauss", "polynomial" etc
        
        if kernel_type == "gauss":
            print("gaussian kernel detected")
            if params == "auto":
                kern = kernel.Gauss_kernel([0])
                self.params = kern.optimal_params(X) #make kernel create its optimal param and return it
            else:
                self.params = params
                kern = kernel.Gauss_kernel(self.params)
            
        elif kernel_type == "polynomial":
            print("polynomial kernel detected")
            if params == "auto":
                kern = kernel.Polynom_kernel([0,0])
                self.params = kern.optimal_params(X)
            else:
                self.params = params
                kern = kernel.Polynom_kernel(self.params)

        elif kernel_type=="identity": #pak pokryt nectverce
            if np.shape(X)[0] != np.shape(X)[1]:
                raise TypeError(f"Kernel matrix must be square matrix, but X has shape {np.shape(X)}")
            else:
                self.K = X.copy()
                self.params = [None]
        else:
            raise NameError(f"kernel type {kernel_type} unknown or not implemented")
            
        for i in range(0, self.n):
            for j in range(i, self.n):
                self.K[i,j] = kern.eval(X[i], X[j])
                self.K[j,i] = self.K[i,j] #symmetric matrix
        
        print("kernel matrix succesfully created")

    def submatrix(self,index_list):
        return self.K[np.ix_(index_list, index_list)]
    
    @classmethod
    def from_matrix(cls, matrix): 
        ret = cls(matrix, 'identity')
        return ret

    def __getitem__(self, position):
        if isinstance(position,tuple):
            a,b = position
            return self.K[a,b]
        elif isinstance(position,list):
            kernel_matrix_ret = self.from_matrix(self.K[np.ix_(position[0],position[1])])
            return kernel_matrix_ret
        
    def __len__(self) -> int:
        return self.n
    
    def __str__(self) -> str:
        return f"Kernel matrix {self.n} x {self.n} \n" + np.array2string(self.K)

        

    
#zamysleni: custom kernel od uzivatele
                
    