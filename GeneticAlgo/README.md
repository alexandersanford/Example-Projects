## Assign 5

### Coding directions

In this fifth assignment we will be implementing a Genetic Algorithm to solve
the Traveling Salesperson problem for a complete, weighted, and directed graph. 

### Required: Function to be implemented

Note that only one function is required for this assignemnt. This is: 


* __TSPwGenAlgo(W, population_size=50, mutation_rate=0.01, explore_rate=0.5)__
  * input: an adjacency weight matrix, maximum number of generations, population
    size to use, mutation rate, and exploration rate (lower exploration rate
    means a smaller group of 'fit' individuals should be used for reproduction,
    a larger exploration rate means a larger group of individuals should be used)
  * output/return: a dictionary containing the solution path and distance, and
    a list with the shortest distance found in each generation (see assign5.py) 

### Optional: Each of these functions implemented (correctly) is +10 pts

* __TSPwDynProg(W)__ 
  * input: adjacency weight matrix, W such that W[i][j] is the weight from vertex i to vertex j
  * output/return: a dictionary containing the solution path and distance (see assign5.py) 

* __TSPwBandB(W)__ carry out Dijkstra's algorithm on a graph represented by its weight matrix
  * input: adjacency weight matrix, W such that W[i][j] is the weight from vertex i to vertex j
  * output/return: a dictionary containing the solution path and distance (see assign5.py) 


Since your program will not be run through unit test on GitHub, there will also
not be any linting checks. However, be sure to document your code, including any
auxiliary functions that you use. You should replace/change the existing
comments that are there to help you get started. 

