import random
import numpy as np
import matplotlib.pyplot as plt # Para plotar a convergência

# --- Parâmetros do Problema ---
N_VARIABLES = 10
LOWER_BOUND = -5.0
UPPER_BOUND = 5.0

# --- Funções do Algoritmo Genético ---

def objective_function(individual):
    """Calcula o valor da função objetivo a ser minimizada."""
    return np.sum(np.square(individual))

def fitness_function(individual):
    """
    Calcula a aptidão do indivíduo.
    Como queremos minimizar f(x), usamos 1 / (1 + f(x)) para que
    valores menores de f(x) resultem em maior aptidão.
    f(x) é sempre >= 0 neste problema.
    """
    obj_value = objective_function(individual)
    # Adicionar uma pequena constante ao denominador para evitar divisão por zero se obj_value for -1
    # Embora para f(x) = sum(x_i^2), obj_value será sempre >= 0.
    return 1.0 / (1.0 + obj_value)

def create_individual():
    """Cria um indivíduo aleatório com N_VARIABLES genes."""
    return [random.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(N_VARIABLES)]

def initialize_population(population_size):
    """Cria a população inicial."""
    return [create_individual() for _ in range(population_size)]

def tournament_selection(population, fitness_values, k=3):
    """
    Seleciona um indivíduo usando seleção por torneio.
    k: tamanho do torneio.
    """
    # Escolhe k índices aleatórios da população (sem reposição)
    tournament_contenders_indices = random.sample(range(len(population)), k)
    
    winner_index = -1
    best_fitness_in_tournament = -float('inf')

    for index in tournament_contenders_indices:
        if fitness_values[index] > best_fitness_in_tournament:
            best_fitness_in_tournament = fitness_values[index]
            winner_index = index
            
    return population[winner_index]

def arithmetic_crossover(parent1, parent2, alpha=0.7):
    """
    Realiza o crossover aritmético entre dois pais.
    Gera um filho. Alpha determina a mistura.
    Um alpha fixo ou aleatório por gene pode ser usado.
    Aqui, usamos um alpha fixo para simplicidade, mas um alpha aleatório por gene
    pode oferecer mais diversidade.
    """
    child = [0.0] * N_VARIABLES
    for i in range(N_VARIABLES):
        # Mistura ponderada dos genes dos pais
        # child[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]

        # Alternativa: BLX-alpha-like para cada gene (como discutido antes)
        # Isto tende a expandir a busca um pouco mais
        gene_p1 = parent1[i]
        gene_p2 = parent2[i]
        
        d = abs(gene_p1 - gene_p2)
        min_val_gene = min(gene_p1, gene_p2) - alpha * d
        max_val_gene = max(gene_p1, gene_p2) + alpha * d
        
        gene_value = random.uniform(min_val_gene, max_val_gene)
        child[i] = np.clip(gene_value, LOWER_BOUND, UPPER_BOUND) # Garantir limites
    return child

def gaussian_mutation(individual, mutation_probability_gene, sigma=0.5):
    """
    Aplica mutação gaussiana a cada gene do indivíduo com uma certa probabilidade.
    mutation_probability_gene: probabilidade de um gene sofrer mutação.
    sigma: desvio padrão da distribuição normal para a mutação.
    """
    mutated_individual = individual[:] # Criar uma cópia para não alterar o original
    for i in range(N_VARIABLES):
        if random.random() < mutation_probability_gene:
            mutation_value = random.gauss(0, sigma) # N(0, sigma)
            mutated_individual[i] += mutation_value
            # Garantir que o gene mutado esteja dentro dos limites
            mutated_individual[i] = np.clip(mutated_individual[i], LOWER_BOUND, UPPER_BOUND)
    return mutated_individual

# --- Loop Principal do Algoritmo Genético ---
def genetic_algorithm(population_size, n_generations, crossover_probability, 
                      mutation_probability_gene, sigma_mutation, tournament_k,
                      use_elitism=True):
    """
    Executa o algoritmo genético.
    """
    # 1. Inicialização
    population = initialize_population(population_size)
    
    best_overall_individual = None
    best_overall_fitness = -float('inf')
    # Para guardar o melhor valor da função objetivo de cada geração
    history_best_obj_value = [] 

    print(f"Iniciando AG: População={population_size}, Gerações={n_generations}, "
          f"CX_Prob={crossover_probability}, Mut_Prob_Gene={mutation_probability_gene}, "
          f"Sigma_Mut={sigma_mutation}, K_Tournament={tournament_k}, Elitismo={use_elitism}\n")

    for generation in range(n_generations):
        # 2. Avaliação (calcular aptidão de todos)
        fitness_values = [fitness_function(ind) for ind in population]

        # Encontrar o melhor indivíduo da geração atual
        current_best_fitness_idx = np.argmax(fitness_values)
        current_best_individual = population[current_best_fitness_idx]
        current_best_fitness = fitness_values[current_best_fitness_idx]
        current_best_obj_value = objective_function(current_best_individual)
        
        history_best_obj_value.append(current_best_obj_value)

        # Atualizar o melhor indivíduo global encontrado até agora
        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_individual = current_best_individual[:] # Copiar

        # 3. Construção da Nova População
        new_population = []

        # Elitismo: Opcional, mas geralmente benéfico.
        # Adicionar o melhor indivíduo da geração atual à nova população.
        if use_elitism:
            new_population.append(current_best_individual[:]) # Copiar

        # Preencher o restante da nova população
        while len(new_population) < population_size:
            # Seleção de pais
            parent1 = tournament_selection(population, fitness_values, k=tournament_k)
            parent2 = tournament_selection(population, fitness_values, k=tournament_k)

            # Crossover
            if random.random() < crossover_probability:
                offspring = arithmetic_crossover(parent1, parent2)
            else:
                # Se não houver crossover, um dos pais pode passar (ou uma cópia deles)
                # Aqui, vamos apenas pegar o pai1 como base para mutação
                # Outra estratégia seria adicionar os dois pais diretamente, se não couberem
                # ambos, escolher aleatoriamente ou o mais apto.
                # Para simplificar, criamos um "offspring" a partir do parent1.
                offspring = parent1[:] 

            # Mutação
            mutated_offspring = gaussian_mutation(offspring, mutation_probability_gene, sigma_mutation)
            
            new_population.append(mutated_offspring)
            
        population = new_population # Atualiza a população para a próxima geração
        
        if (generation + 1) % 10 == 0 or generation == n_generations -1 : # Imprimir progresso
             print(f"Geração {generation + 1}/{n_generations}: "
                   f"Melhor Obj da Geração={current_best_obj_value:.6f}, "
                   f"Melhor Obj Geral={objective_function(best_overall_individual):.6f}")

    final_best_obj_value = objective_function(best_overall_individual)
    print("\n--- Otimização Concluída ---")
    print(f"Melhor indivíduo encontrado: {np.round(best_overall_individual, 4)}")
    print(f"Valor da função objetivo (mínimo): {final_best_obj_value:.8f}")
    print(f"Fitness correspondente: {best_overall_fitness:.8f}")
    
    return best_overall_individual, final_best_obj_value, history_best_obj_value

# --- Execução do Algoritmo e Plotagem ---
if __name__ == "__main__":
    # Hiperparâmetros (VOCÊ DEVE VARIAR ESTES PARA OS SEUS EXPERIMENTOS)
    POP_SIZE = 100                   # Tamanho da população
    N_GENERATIONS = 200              # Número de gerações (critério de parada)
    CROSSOVER_PROB = 0.85            # Probabilidade de ocorrer crossover entre dois pais
    MUTATION_PROB_GENE = 0.05        # Probabilidade de um gene sofrer mutação
    SIGMA_MUTATION = 0.2             # Desvio padrão para a mutação Gaussiana
    K_TOURNAMENT = 3                 # Tamanho do torneio para seleção
    USE_ELITISM = True               # Usar elitismo

    # Executar o AG
    best_solution_vector, best_value, convergence_history = genetic_algorithm(
        population_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        crossover_probability=CROSSOVER_PROB,
        mutation_probability_gene=MUTATION_PROB_GENE,
        sigma_mutation=SIGMA_MUTATION,
        tournament_k=K_TOURNAMENT,
        use_elitism=USE_ELITISM
    )

    # Plotar a convergência
    # (O quão rápido o valor da função objetivo do melhor indivíduo diminui)
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history, label='Melhor $f(x)$ por Geração')
    plt.title(f'Convergência do Algoritmo Genético para $f(x) = \sum x_i^2$')
    plt.xlabel('Geração')
    plt.ylabel('Valor da Função Objetivo do Melhor Indivíduo')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Escala logarítmica pode ser útil se a queda for muito rápida
    plt.show()

    print("\nLembre-se de conduzir experimentos sistemáticos variando os hiperparâmetros,")
    print("justificar suas escolhas de design e analisar os resultados em uma tabela.")