import numpy as np

a = 10
br = 4
bi = 3


def crossover(population, int_vars):
    tampop = np.shape(population)[0]

    curr_pop = population if tampop % 2 == 0 else np.vstack(
        (population, population[np.random.randint(tampop)]))
    pop = np.random.permutation(curr_pop)
    offspring = np.zeros_like(pop)

    for i in range(0, tampop, 2):
        off1 = pop[i]
        off2 = pop[i+1]

        u = np.random.random(len(int_vars))

        for j, int_var in enumerate(int_vars):
            b = bi if int_var else br
            b = b if u[j] > 1/2 else -b

            beta = a + b * np.log(u[j])

            offspring[i, j] = off1[j] + beta * abs(off1[j] - off2[j])
            offspring[i + 1, j] = off2[j] + beta * abs(off1[j] - off2[j])

    return offspring if tampop % 2 == 0 else offspring[:-1]
