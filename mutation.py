import numpy as np
import random

verbose = 0

class Mutation():
    def mutate(self, x):
        pass

    def perturbation(self, x, a, b):
        d = 2*(b-a)
        xi = np.mod(np.array(x)-a, d+.0)
        xi = np.minimum(xi, d-xi)
        xnew = a + xi
        return xnew.astype(int)

    def random_point(self, a, b):
        x = a+np.random.randint(0, b-a+1)-1
        return x
    
    def mut_rstar(self, x, a, b): 
        n = len(x)
        k = np.random.randint(0, n) #n is len of a
        if x[k] == a:
            x[k] = x[k]+1
        elif x[k] == b:
            x[k] = x[k]-1
        else:
            if random.uniform(0,1) < 0.5:
                x[k] = x[k] + 1
            else:
                x[k] = x[k] -1
        return x

class Pareto_mutation(Mutation):
    def __init__(self, a, b, params=[0.1, 1]) -> None:
        if len(params) > 2 | len(params) == 0:
            raise TypeError(f"Pareto mutation takes 2 params(t_mut, alpha), but {len(params)} were given")
        self.t_mut = params[0]
        self.alpha = params[1]
        self.a = a
        self.b = b
        if verbose == 1:
            print(f"Pareto mutation initialized with params = {self.t_mut} and {self.alpha}")
    def mutate(self, x): 
        n = len(x)
        eta = np.random.normal(0,1,n)
        delta = np.power(np.random.uniform(0, 1), -1 / self.alpha)
        xi = eta / np.linalg.norm(eta) * delta
        xtrial = np.floor(x + self.t_mut * xi + 0.5)
        xnew = self.perturbation(xtrial, 0, self.b)
        if np.all(xnew == x): 
            x = self.mut_rstar(x, self.a, self.b) # generate new x using r-star mutation
        else:
            for i in range(0, len(xnew)):
                x[i] = xnew[i]
    
class Wild_mutation(Mutation):
    def __init__(self, a, b, params = [0.1]) -> None:
        
        if len(params) != 1:
            raise TypeError(f"Wild mutation takes 1 param(pmut), but {len(params)} were given")
        self.pwild = params[0]
        self.a = a
        self.b = b
        if verbose == 1:
            print(f"Wild mutation initialized with param {self.pwild}")
    def mutate(self, x):
        if random.uniform(0,1)<self.pwild:
            x = np.random.randint(self.a, self.b+1, len(x)) #generate random new x
        else:
            x = self.mut_rstar(x, self.a, self.b) #use rstar mutation to mutate

class Hamming_mutation(Mutation):
    def __init__(self, a, b, params = [1]) -> None:
        if len(params) != 1:
            raise TypeError(f"Hamming mutation takes 1 param(nmut), but {len(params)} were given")
        self.nmut = params[0]
        self.a = a
        self.b = b
        if verbose==1:
            print(f"Hamming mutation initialized with param = {self.nmut}")
    def mutate(self, x):
        n = len(x)
        for _ in range(0, self.nmut):
            i = random.randint(0, n-1)  # Choose a random index to mutate
            old_value = x[i]
            new_value = old_value
            while new_value == old_value:
                new_value = (old_value + random.randint(0, (self.b+1))) % (self.b+1)  # Choose a new value that is different from the old value
            x[i] = new_value
        