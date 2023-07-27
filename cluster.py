import numpy as np
import mutation as mut

verbose = 0

class KMeans():
    def __init__(self, n_clusters, max_iter=100) -> None: 
        self.n_clusters = n_clusters
        self.max_iter = max_iter #max iterations for one kmeans run
        if verbose == 1:
            print("K Means initialized.")


    def fit_one(self, K): #one run of kmeans
        self.partition_vector = np.random.randint(0, self.n_clusters, K.n) #initialize random partition
        if verbose == 1:
            print(f"starting objective function:{self.objective_function(K, self.partition_vector)}")
        for _ in range(0, self.max_iter):
            #pre-count all cluster submatrices and cluster indexes
            cluster_submatrices = []
            cluster_indexes = []
            for c in range(0, self.n_clusters):
                index_list = [ i for i in range(len(self.partition_vector)) if self.partition_vector[i] == c]
                cluster_indexes.append(index_list)
                cluster_submatrices.append(K.submatrix(index_list))

            #allocate distance matrix, which will hold distance of every point from every cluster centroid
            distance_matrix = np.zeros((K.n, self.n_clusters))
            #fill distance matrix
            for x_i in range(0, K.n):
                for c in range(0, self.n_clusters):
                    if len(cluster_indexes[c]) == 0: #empty cluster->abort
                        raise ValueError("Empty cluster found, restart the algorithm.")
                    
                    small_sum = np.sum([K[x_i, x_l] for x_l in cluster_indexes[c]]) 
                    sum = K[x_i, x_i] + np.mean(cluster_submatrices[c]) -2 * small_sum / len(cluster_indexes[c])
                    distance_matrix[x_i, c] = sum
            
            #choose the minimal distance for every x_i
            partition_new = np.argmin(distance_matrix, axis=1)

            if np.array_equal(self.partition_vector,partition_new): #if non of the elements moved
                break
            else:
                self.partition_vector = partition_new

        if verbose == 1:
            print(f"ending objective function:{self.objective_function(K, self.partition_vector)}")
        return self.partition_vector, self.objective_function(K, self.partition_vector)

    def fit(self, K, n):
        # fit n times k-means and choose the best result
        if verbose == 1:
            print(f"Starting K Means fit {n} times. The best one will be chosen at the end.")
        parts = []
        objs = []
        for _ in range(n):
            part, obj = self.fit_one(K)
            parts.append(part)
            objs.append(obj)
        index = np.argmin(objs)
        return parts[index], objs[index]


    def objective_function(self,K,partition): 
        obj = 0 
        for c in range(0, self.n_clusters):
            index_list = [i for i in range(len(partition)) if partition[i] == c] #index list of elements belonging to cluster c
            if index_list:
                K_sub = K.submatrix(index_list)
                obj += np.trace(K_sub) - np.mean(K_sub)*len(index_list) 
        return obj
    
class Heuristics(): #mozna to pojmenovat random descent?
    def __init__(self, n_clusters, max_iter=10000) -> None: 
        self.n_clusters = n_clusters
        self.max_iter = max_iter #max iterations of random descent
        if verbose == 1:
            print("Random descent initialized.")

        self.mutation_dic =  {
            "Pareto": mut.Pareto_mutation,
            "Hamming": mut.Hamming_mutation,
            "Wild": mut.Wild_mutation
        }

    def objective_function(self,K,partition): 
        obj = 0 
        for c in range(0, self.n_clusters):
            index_list = [i for i in range(len(partition)) if partition[i] == c] #index list of elements belonging to cluster c
            if index_list:
                K_sub = K.submatrix(index_list)
                obj += np.trace(K_sub) - np.mean(K_sub)*len(index_list) 
        return obj

    def fit(self, K, mutation, params = "default"):
        """
        Perform Random Descent with mutation on the given function f, starting from the initial solution x_init,
        for at most max_iter iterations.
        """
        if verbose == 1:
            print(f"starting random descent with {mutation} mutation.")
        self.partition_vector = np.random.randint(0, self.n_clusters, K.n)

        x_best = self.partition_vector
        f_best = self.objective_function(K, x_best)
        if verbose == 1:
            print(f"f starting: {f_best}")

        if params == "default":
            mut_curr = self.mutation_dic[mutation](0, self.n_clusters-1) 
        else:
            mut_curr = self.mutation_dic[mutation](0, self.n_clusters-1, params)

        for _ in range(0, self.max_iter):

            x_new = x_best.copy()
            mut_curr.mutate(x_new)
            f_new = self.objective_function(K, x_new)
            if f_new < f_best:
                x_best = x_new
                self.partition_vector = x_best
                f_best = f_new
        return self.partition_vector, f_best
    
     
    
    