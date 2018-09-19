from __future__ import print_function
import itertools as it
from random import sample, random, randint
import numpy as np
#from dp import jumpIt, displayPath



global cost, path

cost = [] # global table to cache results - cost[i] stores minimum cost of playing the game starting at cell i
path = [] #global table to store path leading to cheapest cost
def jumpIt(board):
    #Bottom up dynamic programming implementation
    #board - list with cost associated with visiting each cell
    #return minimum total cost of playing game starting at cell 0
    
    n = len(board)
    cost[n - 1] = board[n - 1] #cost if starting at last cell
    path[n - 1] = -1 # special marker indicating end of path "destination/last cell reached"
    cost[n - 2] = board[n - 2] + board[n - 1] #cost if starting at cell before last cell
    path[n -2] = n - 1 #from cell before last, move into last cell
    #now fill the rest of the table
    for i in range(n-3, -1, -1):
        #cost[i] = board[i] + min(cost[i+1], cost[i+2])
        if cost[i +  1] < cost[i + 2]: # case it is cheaper to move to adjacent cell
            cost[i] = board[i] +  cost[i + 1]
            path[i] = i + 1 #so from cell i, one moves to adjacent cell
        else: 
            cost[i] = board[i] + cost[i + 2]
            path[i] = i + 2 #so from cell i, one jumps over cell
    return cost[0]

def displayPath(board):
    #Display path leading to cheapest cost - method displays indices of cells visited
    #path - global list where path[i] indicates the cell to move to from cell i
    cell = 0 # start path at cell 0
    print("path showing indices of visited cells:", end = " ")
    print(0, end ="")
    path_contents = "0" # cost of starting/1st cell is 0; used for easier tracing
    while path[cell] != -1: # -1 indicates that destination/last cell has been reached
        print(" ->", path[cell], end = "")
        cell = path[cell]
        path_contents += " -> " + str(board[cell])
    print()
    print("path showing contents of visited cells:", path_contents)

def initialize_population(board_len, pop_size):
	pop = np.random.randint(0,2,size=board_len*pop_size)
	pop = pop.reshape(pop_size, board_len)
	pop = clear_double_zeros(pop)

	while len(pop) < pop_size:
		num_to_add = pop_size - len(pop)
		d = np.array(np.random.randint(0,2,size=board_len*num_to_add), dtype=int).reshape(num_to_add, board_len)
		d = clear_double_zeros(d)

		if len(d) != 0:
			#pop = np.concatenate((pop, d), axis=0)
			pop = np.vstack((pop,d))

	return pop


def clear_double_zeros(population):
	pop = np.array(population.copy(), dtype=int)
	non_dub_indices = np.array([], dtype=int)

	for j,chrom in enumerate(population):
		double = False
		for i in range(len(chrom)-1):
			if chrom[i] == chrom[i+1] and chrom[i] == 0:
				double = True

		if not double:
			non_dub_indices = np.append(non_dub_indices, j)

	pop = pop[non_dub_indices]

	return pop


def get_probs(pop, board):
	# Roulette wheel selection.
	pop_fitness = np.array([fitness(route, board) for route in pop], dtype=np.float64)
	tot_fit = np.sum(pop_fitness)
	probs = np.array([(pop_fitness[i]/tot_fit) for i in range(len(pop_fitness))])

	return probs, pop_fitness


def prob_select(probs):
	rv = random()
	for i in range(len(probs)):
		rv -= probs[i]
		if rv < 0.0:
			return i

	return len(probs)-1


def mutate(c1, c2):
	i = randint(0, len(c1)-1)

	c1[i] = (c1[i]*-1) + 1
	c2[i] = (c2[i]*-1) + 1

	return c1, c2




def crossover(p1, p2):
	p1, p2, = list(p1), list(p2)
	locus = randint(0, len(p1)-1)

	child1 = np.array(p1[:locus]+p2[locus:])
	child2 = np.array(p2[:locus]+p1[locus:])

	return child1, child2



def genetic(game_board, pop_size, max_gens, crossover_prob, mutation_prob):
	board = game_board[1:-1]
	n = len(board)
	pop = initialize_population(n, pop_size)
	
	for g in range(max_gens):
		probs, pop_fitness = get_probs(pop, board)

		i = pop_fitness.argsort()
		pop = pop[i][:-2, :]
		probs = probs[i][:-2]

		parent1 = pop[prob_select(probs)]
		parent2 = pop[prob_select(probs)]

		c = random()
		(child1, child2) = crossover(parent1, parent2) if c <= crossover_prob else (parent1, parent2)

		m =random()
		child1, child2 = mutate(child1, child2) if m <= mutation_prob else child1, child2

		pop = np.vstack((pop, child1, child2))  		
		pop = clear_double_zeros(pop)						

	pop_costs = np.array([np.dot(route, board) for route in pop])

	mc = np.argmin(pop_costs)

	path = pop[mc]
	path = np.hstack((0,path,1))

	min_cost = np.dot(path, game_board)

	path_contents = np.multiply(path, game_board)
	path_contents = path_contents[np.nonzero(path_contents)]

	path = np.multiply(path, np.arange(n+2))
	path = list(path[np.nonzero(path)])

	return min_cost, path, path_contents



def fitness(route, board):
	dist = np.dot(route, board)

	return dist



def ga_path(board, path, contents):
    path.append(-1)
    print("path showing indices of visited cells:", end = " ")
    print(0, end ="")
    path_contents = "0" # cost of starting/1st cell is 0; used for easier tracing
    i = 0
    while path[i] != -1: # -1 indicates that destination/last cell has been reached
        print(" ->", path[i], end = "")
        path_contents += " -> " + str(contents[i])
        i += 1
    print()
    print("path showing contents of visited cells:", path_contents)


def main():
    f = open("input2.txt", "r") #input.txt
    global cost, path
    max_gens = 500

    crossover_prob = 0.90
    mutation_prob  = 0.05

    pop_size = 150

    missed = 0.0
    correct = 0.0

    for line in f:
        lyst = line.split() # tokenize input line, it also removes EOL marker
        lyst = list(map(int, lyst))
        cost = [0] * len(lyst) #create the cache table
        path = cost[:] # create a table for path that is identical to path
        print("game board: ", lyst)
        print("______________________________________")
        print("DP Solution")
        minimum_cost = jumpIt(lyst)
        print("Minimum Cost: ", minimum_cost)
        print()
        displayPath(lyst)
        print("______________________________________")
        print("GA Solution")
        (min_cost, path, path_contents) = genetic(lyst, pop_size, max_gens, crossover_prob, mutation_prob)
        print("Minimum Cost (fitness): ", min_cost)
        ga_path(lyst, path, path_contents)
        print("================================================================\n")
        if min_cost == minimum_cost:
        	correct += 1.0
        else:
        	missed += 1.0

    accuracy = round(correct/(missed+correct), 4) * 100.0
    print('Accuracy is {}%'.format(accuracy))

if __name__ == "__main__":
    main()

