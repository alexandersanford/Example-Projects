"""
For this assignment there is no automated testing. You will instead submit
your *.py file in Canvas. I will download and test your program from Canvas.
Student name: Alex Sanford
Online resources used: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
Note on above: totally different structure for managing cities and distances, but gave me some ideas on how to crossover and how to select parents.
Also consulted with Nicholas Ryan; he gave me some pointers on getting started.
High score so far is 249 on complete_graph_n100.txt! max_num_generations = 10000, population_size = 1000, mutation_rate = 0.27, explore rate = 0.5
Not sure why, but my algorithm works far better on large graphs than it does on small graphs. Tends to flatline rather quickly on the small graphs.
High mutation_rate => high risk; I've gotten a better solution overall, but it leads to far more failures and a much less linear evolution.
This was a fun assignment! Thanks Dr. G!
"""

import sys
import random
import time

def adjMatFromFile(filename):
    """ Create an adj/weight matrix from a file with verts, neighbors, and weights. """
    f = open(filename, "r")
    n_verts = int(f.readline())
    print(f" n_verts = {n_verts}")
    adjmat = [[None] * n_verts for i in range(n_verts)]
    for i in range(n_verts):
        adjmat[i][i] = 0
    for line in f:
        int_list = [int(i) for i in line.split()]
        vert = int_list.pop(0)
        assert len(int_list) % 2 == 0
        n_neighbors = len(int_list) // 2
        neighbors = [int_list[n] for n in range(0, len(int_list), 2)]
        distances = [int_list[d] for d in range(1, len(int_list), 2)]
        for i in range(n_neighbors):
            adjmat[vert][neighbors[i]] = distances[i]
    f.close()
    return adjmat


def printAdjMat(mat, width=3):
    """ Print an adj/weight matrix padded with spaces and with vertex names. """
    res_str = '     ' + ' '.join([str(v).rjust(width, ' ') for v in range(len(mat))]) + '\n'
    res_str += '    ' + '-' * ((width + 1) * len(mat)) + '\n'
    for i, row in enumerate(mat):
        row_str = [str(elem).rjust(width, ' ') for elem in row]
        res_str += ' ' + str(i).rjust(2, ' ') + ' |' + ' '.join(row_str) + "\n"
    print(res_str)


def routeDistance(route, g):
    """ Determine total cost of route given that the final vertex will continue on to first vertex """
    distance = 0
    for i in range(len(route)):
        if i == len(route) - 1:
            distance += g[route[i]][route[0]]
        else:
            distance += g[route[i]][route[i+1]]
    return distance


def mutate(individual):
    """ Function to mutate an individual by swapping two random vertices within the individual"""
    # choose x and y randomly within the length of the individual
    x = int(random.random() * (len(individual) - 1))
    y = int(random.random() * (len(individual) - 1))
    if x == y:
        y += 1
    individual[x], individual[y] = individual[y], individual[x]


def breed(parentOne, parentTwo):
    """ Function that produces offspring based on components of each parent """
    children = [None, None]
    stop = int(len(parentOne) / 2)
    children[0] = parentOne[:stop]
    for i in range(len(parentTwo)):
        if parentTwo[i] not in children[0]:
            children[0].append(parentTwo[i])
    children[1] = parentTwo[:stop]
    for i in range(len(parentOne)):
        if parentOne[i] not in children[1]:
            children[1].append(parentOne[i])
    return children


def TSPwGenAlgo(g, max_num_generations=100, population_size=50,
        mutation_rate=0.25, explore_rate=0.5):
    """ A genetic algorithm to attempt to find an optimal solution to TSP  """
    solution_cycle_distance = None # the distance of the final solution cycle/path
    solution_cycle_path = [] # the sequence of vertices representing final sol path to be returned
    shortest_path_each_generation = [] # store shortest path found in each generation

    best = sys.maxsize # keep track of the best route overall to see if algorithm is improving

    failures = 0 # number of times that the algorithm goes backwards
    failstop = False # keep track of whether the number of failures triggered the early-stop condition

    sames = -1 # number of times that the algorithm generates the same best path as the previous generation(s)
    samestop = False # keep track of whether the number of sames triggered the early-stop condition

    # Create a population of specified size, give each individual a copy of the 'alphabet' of vertices
    # and set population_size and max_num_generations to values that are appropriate for the graph size
    vertices = [i for i in range(len(g))]
    population_size = len(vertices) * 10
    max_num_generations = population_size * 10
    population = [i for i in range(population_size)]
    
    # initialize individuals to an initial random 'solution'
    for i in range(population_size):
        individual = vertices.copy()
        random.shuffle(individual)
        population[i] = individual
    
    # loop for x number of generations (with possibly other early-stopping criteria)
    for x in range(max_num_generations):
        # Early stop: if the algorithm depreciates in quality enough times, the evolution will stop
        if failures > (max_num_generations / 45): # note to self: 45 was chosen through trial and error
            failstop = True
            break

        # Early stop: if the algorithm has plateaued for a significant series of generations, the evolution will stop
        if sames > (max_num_generations / 45): # note to self: 45 was chosen through trial and error
            samestop = True
            break

        # track number of consecutive identical bests
        sames += 1

        # calculate fitness of each individual in the population
        distances = [0 * len(population) for i in range(len(population))]
        for i in range(len(distances)):
            distances[i] = routeDistance(population[i],g)

        # sort individuals by distance from lowest to highest
        ranked = sorted(list(zip(distances, population)))
        
        # if the value has updated beyond the previous best, reset consecutive identical bests
        if ranked[0][0] != best:
            sames = 0
        
        # if the best value this generation is better than the all-time, update all-time best
        if ranked[0][0] < best:
            best = ranked[0][0]
            solution_cycle_distance = best
            solution_cycle_path = ranked[0][1]
        
        # if the best value this generation is worse than the all-time best, mark this as a failure (backwards progress)
        elif ranked[0][0] > best:
            failures += 1
        
        # append distance of the 'fittest' to shortest_path_each_generation)
        shortest_path_each_generation.append(ranked[0][0])

        # select the individuals to be used to spawn the generation
        parents = []
        for i in range(int(population_size*explore_rate)):
            parents.append(ranked[i])

        # prevent identical individuals from becoming parents of the next generation
        i = 0
        while (i < len(parents) - 1):
            if parents[i][1] == parents[i+1][1]:
                parents.remove(parents[i+1])
            else:
                i += 1
        
        # initialize empty population
        population = []

        # append top 10% of the parents to the next generation ('elitism')
        for i in range(int(len(parents) / 10)):
            population.append(parents[i][1])

        # populate through breeding parents
        while len(population) < population_size:
            # choose x and y through some kind of random procedure between 0 and len(parents)
            x = int(random.random() * (len(parents) - 1))
            y = int(random.random() * (len(parents) - 1))
            children = breed(parents[x][1], parents[y][1]) # produces an array of two children each time breed() is called
            if len(population) < population_size:
                population.append(children[0])
            if len(population) < population_size:
                population.append(children[1])
        
        # allow for mutations (this should not happen too often)
        # Per some random value from 0 to 1, if it is smaller than mutation_rate, mutate that member
        # Repeat for each member in the population
        for i in range(len(population)):
            if random.random() < mutation_rate:
                mutate(population[i])
        
        # diagnostic printing
        print(f"g:{len(shortest_path_each_generation)} b:{ranked[0][0]} s:{sames} f:{failures}")

    # check whether an early stop condition was triggered, and print according data; if none were, show the total number of generations.
    print(shortest_path_each_generation)
    if (failstop):
        print(f"Algorithm stopped reproducing after {failures} failures, generating a total of {len(shortest_path_each_generation)} generations.")
    elif (samestop):
        print(f"Algorithm stopped reproducing after {sames} consecutive identical bests, generating a total of {len(shortest_path_each_generation)} generations.")
    else: 
        print(f"No early stop detected. Number of generations: {len(shortest_path_each_generation)}")

    #return the all-time best solution generated by this algorithm
    return {
            'solution': solution_cycle_path,
            'solution_distance': solution_cycle_distance,
            'evolution': shortest_path_each_generation
           }


def TSPwDynProg(g):
    """ (10pts extra credit) A dynamic programming approach to solve TSP """
    solution_cycle_distance = None # the distance of the final solution cycle/path
    solution_cycle_path = [] # the sequence of vertices representing final sol path to be returned

    #...

    return {
            'solution': solution_cycle_path,
            'solution_distance': solution_cycle_distance,
           }


def TSPwBandB(g):
    """ (10pts extra credit) A branch and bound approach to solve TSP """
    solution_cycle_distance = None # the distance of the final solution cycle/path
    solution_cycle_path = [] # the sequence of vertices representing final sol path to be returned

    #...

    return {
            'solution': solution_cycle_path,
            'solution_distance': solution_cycle_distance,
           }


def assign05_main():
    """ Load the graph (change the filename when you're ready to test larger ones) """
    g = adjMatFromFile("complete_graph_n100.txt")
    printAdjMat(g) # wanted to see the graph to wrap my head around getting started

    # Run genetic algorithm to find plausible and (hopefully) somewhat optimized solution
    start_time = time.time()
    res_ga = TSPwGenAlgo(g)
    elapsed_time_ga = time.time() - start_time
    print(f"GenAlgo runtime: {elapsed_time_ga:.2f}")
    print(f"  sol dist: {res_ga['solution_distance']}")
    print(f"  sol path: {res_ga['solution']}")

    # (Try to) run Dynamic Programming algorithm only when n_verts <= 10
    if len(g) <= 10:
        start_time = time.time()
        res_dyn_prog = TSPwDynProg(g)
        elapsed_time = time.time() - start_time
        if len(res_dyn_prog['solution']) == len(g) + 1:
            print(f"Dyn Prog runtime: {elapsed_time:.2f}")
            print(f"  sol dist: {res_dyn_prog['solution_distance']}")
            print(f"  sol path: {res_dyn_prog['solution']}")

    # (Try to) run Branch and Bound only when n_verts <= 10
    if len(g) <= 10:
        start_time = time.time()
        res_bnb = TSPwBandB(g)
        elapsed_time = time.time() - start_time
        if len(res_bnb['solution']) == len(g) + 1:
            print(f"Branch & Bound runtime: {elapsed_time:.2f}")
            print(f"  sol dist: {res_bnb['solution_distance']}")
            print(f"  sol path: {res_bnb['solution']}")


# Check if the program is being run directly (i.e. not being imported)
if __name__ == '__main__':
    assign05_main()
