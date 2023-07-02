import numpy as np
import random

class Mutation():
    def mutate(self, x):
        pass

    def perturbation(self, xtrial, a, b):
        #print(f"\nstarting perturbation of:{xtrial}")
        d = 2*(b-a)
        xi = np.mod(np.array(xtrial)-a, d+.0)
        xi = np.minimum(xi, d-xi)
        xnew = a + xi
        return xnew.astype(int)

    def random_point(self, a, b):
        x = a+np.random.randint(0, b-a+1)-1
        return x
    
    def mut_rstar(self, x, a, b):
        n = len(x)
        k = np.random.randint(0, n) #n je delka a
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
    def __init__(self, a, b, params=[0.1, 1], verbose=0) -> None:
        if len(params) > 2 | len(params) == 0:
            print(f"Pareto mutation takes 2 params(t_mut, alpha), but {len(params)} were given")
        self.t_mut = params[0]
        self.alpha = params[1]
        self.a = a
        self.b = b
        print(f"Pareto mutation initialized with params = {self.t_mut} and {self.alpha}")
    def mutate(self, x): #vstupni parameter dostane funkce, chci ho upravit
        n = len(x)
        eta = np.random.normal(0,1,n)
        delta = np.power(np.random.uniform(0, 1), -1 / self.alpha)
        xi = eta / np.linalg.norm(eta) * delta
        xtrial = np.floor(x + self.t_mut * xi + 0.5)
        xnew = self.perturbation(xtrial, 0, self.b)
        if np.all(xnew == x): #pokud se dostanu sem
            #print("nothing")
            x = self.mut_rstar(x, self.a, self.b) #vygeneruju nove x jinou funkci -> funguje dobre
        else:
            #jinak chci do toho x priradit to nove xnew
            #print("something")
            #print(f"xnew: {xnew}")
            #toto nefunguje
            for i in range(0, len(xnew)):
                x[i] = xnew[i]
            #ani toto nefunguje
            #x = xnew

            #vzdy to vrati to puvodni x co ta funkce dostala
    
class Wild_mutation(Mutation):
    def __init__(self, a, b, params = [0.1], verbose=0) -> None:
        
        if len(params) != 1:
            print(f"Wild mutation takes 1 param(pmut), but {len(params)} were given")
        self.pwild = params[0]
        self.a = a
        self.b = b
        print(f"Starting wild mutation with param {self.pwild}")
    def mutate(self, x):
        #print(f"starting mutation of X: {x}")
        if random.uniform(0,1)<self.pwild:
            #print("wild")
            x = np.random.randint(self.a, self.b+1, len(x))
        else:
            #print("not wild")
            x = self.mut_rstar(x, self.a, self.b)
        #print(f"mutated: {x}")

class Hamming_mutation(Mutation):
    def __init__(self, a, b, params = [1], verbose=0) -> None:
        self.nmut = params[0]
        self.a = a
        self.b = b
        #if self.verbose==1:
        print(f"Hamming mutation initialized with param = {self.nmut}")
    def mutate(self, x):
        n = len(x)
        for j in range(0, self.nmut):
            i = random.randint(0, n-1)  # Choose a random index to mutate
            old_value = x[i]
            new_value = old_value
            while new_value == old_value:
                new_value = (old_value + random.randint(0, (self.b+1))) % (self.b+1)  # Choose a new value that is different from the old value
            x[i] = new_value
            #print(f"xnew after {j} iter:\n{xnew}")
        