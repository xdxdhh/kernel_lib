import numpy as np



class Kernel_base:
    def eval(self, x, y):
        pass
    def optimal_params(self, X):
        raise NotImplementedError

class Gauss_kernel(Kernel_base): 
    def __init__(self, params):
        if len(params) == 0 | len(params) > 1:
            raise TypeError(f"Gaussian kernel takes 1 parameter, but {len(params)} were given.") 
        else:
            print("Gaussian kernel initialized") #should be only in verbose mode again?
            self.sigma = params[0]

    def eval(self, x, y):
        if len(x) != len(y):
            raise ValueError(f"x and y must be of the same length, len x:{len(x)}, len y:{len(y)}")
        diff = x - y
        squared_norm = sum(x_i**2 for x_i in diff)
        return np.exp(-squared_norm / (2 * self.sigma**2))
    
    def optimal_params(self, X):
        data_num = np.shape(X)[0]
        squared_diff_list = []
        for i in range(0, data_num):
            for j in range(0,data_num):
                if i!=j:
                    diff = X[i] - X[j]
                    squared_diff_list.append(sum(x_i**2 for x_i in diff)) 
                    
        q9 = np.quantile(squared_diff_list, 0.9)
        q1 = np.quantile(squared_diff_list, 0.1)
        sigma = np.sqrt(np.mean([q1, q9]))
        self.sigma = np.round(sigma, 3)
        print(f"optimal sigma param found: {np.round(sigma, 3)}")
        return [self.sigma]
    
class Polynom_kernel(Kernel_base):
    def __init__(self, params): #params:degree d, shift c 
        if len(params) == 1: #only degree, c is zero
            self.degree = params[0]
            self.c = None
            print("Polynomial kernel initialized")
        elif len(params) == 2:
            self.degree = params[0]
            self.c = params[1]
            print("Polynomial kernel initialized")
        else:
            raise TypeError(f"Polynomial kernel takes 1 parameter(d), or 2 parameters(d,c), but {len(params)} were given.")

    def eval(self, x, y):
        if len(x) != len(y):
                raise ValueError(f"x and y must be of the same length, len x:{len(x)}, len y:{len(y)}")
        if self.c == None:
            return np.power(np.dot(x,y), self.degree)
        else:
            if len(self.c) != len(x):
                raise ValueError(f"input vectors and c must be of the same length, len x:{len(x)}, len c:{len(self.c)}")
            return np.power(np.dot(x,y) + self.c, self.degree)



