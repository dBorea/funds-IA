import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import csv


# INITIALIZING VARIABLES
obj_prices = []
obj_weights = []
with open('mochila.txt', 'r') as fin:
    for i, row in enumerate(csv.reader(fin)):
        if i == 1:
            n_obj = int(row[0])
        if i == 2:
            max_load = int(row[0])
        if i > 3:
            p, w = str.split(row.pop(),' ')
            obj_prices.append(int(p))
            obj_weights.append(int(w))

genome_size = n_obj
ideal_fitness = sum(obj_prices)

# FITNESS FUNCTION
def fit_backpack(gene, weights, prices):
    if(np.dot(gene, weights) <= max_load):
        return np.dot(gene, prices)
    return 0

# EXECUTION FUNCTION
def fill_knapsack(pop_size, gen_limit, mutation_rate, actv_logs=False):

    # EXECUTION
    population = rnd.randint(low=0, high=2, size=(pop_size, genome_size))
    best_fit_per_gen = np.zeros(gen_limit)

    for gen_number in range(0, gen_limit):

        # fitness scoring
        fit_scores = []
        for i in range(0, pop_size):
            fit_scores.append(fit_backpack(population[i], obj_weights, obj_prices))
        
        # best scores
        best_fit_per_gen[gen_number] = max(fit_scores)
        best_fit_idx = np.argmax(fit_scores)
        best_genome = population[best_fit_idx]

        # end criteria
        if best_fit_per_gen[gen_number] >= ideal_fitness:
            break

        # parent selection
        parents = np.zeros(shape=(pop_size, genome_size))

        for i in range(0, pop_size):
            idx_first_parent = rnd.randint(pop_size)
            idx_second_parent = rnd.randint(pop_size)
            if fit_scores[idx_first_parent] > fit_scores[idx_second_parent]:
                parents[i] = population[idx_first_parent]
            else:
                parents[i] = population[idx_second_parent]

        # crossover
        offspring = np.zeros(shape=(pop_size, genome_size))
        for i in range(0, pop_size):
            crossover_point = rnd.randint(genome_size - 1)
            offspring[i][0:crossover_point] = parents[i][0:crossover_point]
            offspring[i][crossover_point+1:genome_size-1] = parents[i][crossover_point+1:genome_size-1]

        # mutation
        for i in range(0, pop_size):
            for j in range(0, genome_size):
                if rnd.random() < mutation_rate:
                    offspring[i][j] = int(not(offspring[i][j]))

        # new population
        population = offspring
        population[0] = best_genome

    if actv_logs:
        print("Melhor genoma: \n", best_genome)
        print("Valor: \n", np.dot(best_genome, obj_prices), " do total ", ideal_fitness)
        print("Peso: \n", np.dot(best_genome, obj_weights), " do máximo ", max_load)
        print("Geração do melhor resultado: \n", np.argmax(best_fit_per_gen))

    return (best_fit_per_gen, best_genome)


# main code

# default values are 100, 2000 and 0.05
pop_samples = [10, 50, 100, 200, 1000]
limit_samples = [100, 500, 1000, 2000, 5000]
mutation_samples = [0.01, 0.05, 0.1, 0.2, 0.5]

varying_pop_results = []
varying_limit_results = []
varying_mutation_results = []

n_iter = 5

for i in range(0, n_iter):
    varying_pop_results.append(fill_knapsack(pop_samples[i], 2000, 0.05, actv_logs=True))
    varying_limit_results.append(fill_knapsack(100, limit_samples[i], 0.05, actv_logs=True))
    varying_mutation_results.append(fill_knapsack(100, 2000, mutation_samples[i]))

fig1 = plt.figure(1)
plt.xlabel('Geração')
plt.ylabel('Distância até a fitness ideal')
plt.title('Variação do tamanho da população')
for i in range(0, n_iter):
    plt.plot(ideal_fitness - varying_pop_results[i][0], label="População de tamanho "+str(pop_samples[i]))
plt.legend(loc="upper right")
plt.grid(True)
fig1.show()

fig2 = plt.figure(2)
plt.xlabel('Geração')
plt.ylabel('Distância até a fitness ideal')
plt.title('Variação do tamanho da população')
for i in range(0, n_iter):
    plt.plot(ideal_fitness - varying_limit_results[i][0], label="Limite de "+str(limit_samples[i])+" gerações")
plt.legend(loc="upper right")
plt.grid(True)
fig2.show()

fig3 = plt.figure(3)
plt.xlabel('Geração')
plt.ylabel('Distância até a fitness ideal')
plt.title('Variação do tamanho da população')
for i in range(0, n_iter):
    plt.plot(ideal_fitness - varying_mutation_results[i][0], label="Mutação em "+str(mutation_samples[i]*100)+"%")
plt.legend(loc="upper right")
plt.grid(True)
fig3.show()

input()
fig1.savefig("plot_1.png")
fig2.savefig("plot_2.png")
fig3.savefig("plot_3.png")