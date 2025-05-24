import numpy as np
import numpy.random as rnd

num_variables = 10
var_bounds = (-5, 5)
pop_size = 100
gen_limit = 200
mutation_rate = 0.05
crossover_rate = 0.85
sigma = 0.2

def fitness(ind):
    return np.sum(ind ** 2)

population = rnd.uniform(var_bounds[0], var_bounds[1], size=(pop_size, num_variables))
best_fit_per_gen = np.zeros(gen_limit)

for gen in range(gen_limit):
    fit_scores = np.array([fitness(ind) for ind in population])
    
    best_idx = np.argmin(fit_scores)
    best_individual = population[best_idx].copy()
    best_fit_per_gen[gen] = fit_scores[best_idx]

    parents = []
    for _ in range(pop_size):
        i1, i2 = rnd.choice(pop_size, size=2, replace=False)
        selected = population[i1] if fit_scores[i1] < fit_scores[i2] else population[i2]
        parents.append(selected)
    parents = np.array(parents)

    offspring = []
    for i in range(0, pop_size, 2):
        p1 = parents[i]
        p2 = parents[i + 1 if i + 1 < pop_size else 0]
        if rnd.random() < crossover_rate:
            alpha = rnd.rand()
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = (1 - alpha) * p1 + alpha * p2
            offspring.extend([c1, c2])
        else:
            offspring.extend([p1.copy(), p2.copy()])
    offspring = np.array(offspring[:pop_size])

    for i in range(pop_size):
        for j in range(num_variables):
            if rnd.random() < mutation_rate:
                offspring[i][j] += rnd.normal(0, sigma)
                offspring[i][j] = np.clip(offspring[i][j], var_bounds[0], var_bounds[1])

    offspring[0] = best_individual
    population = offspring

print("Melhor indivíduo:", best_individual)
print("Fitness:", fitness(best_individual))
print("Geração do melhor resultado:", np.argmin(best_fit_per_gen))
