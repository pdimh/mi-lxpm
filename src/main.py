import importlib
import laplace_crossover as lx
import power_mutation as pm
import numpy as np
import tournament
import truncation as tr
import utils

config = utils.config
p = importlib.import_module(f'problems.{config.problem}')

pop_size = config.pop_factor * len(p.INT_VARS)
elite_size = int(np.ceil(config.elite_factor * pop_size))

offspring = utils.generate(pop_size, p.INT_VARS, p.L_BOUND, p.U_BOUND)
score, feasible = utils.evaluate(offspring, p.evaluate, p.MAX)

fittest = None
f_best = None
success = None

for i in range(0, config.max_iterations):
    score_sort = np.argsort(score) if not p.MAX else np.argsort(score)[::-1]

    if feasible and (fittest is None or (p.MAX and f_best < score[score_sort[0]]) or (not p.MAX and f_best > score[score_sort[0]])):
        fittest = offspring[score_sort[0]]
        f_best = score[score_sort[0]]
        if hasattr(p, 'OPTIMAL'):
            if p.OPTIMAL == 0 and np.isclose(f_best, p.OPTIMAL, atol=0.01):
                success = True
                break
            elif np.isclose(f_best, p.OPTIMAL, rtol=0.01):
                success = True
                break

    elite = offspring[score_sort][0:elite_size]
    selection = tournament.select(
        offspring, score, pop_size - elite_size, p.MAX)

    n_cross = pop_size - elite_size
    off = lx.crossover(selection[0:n_cross], p.INT_VARS)

    off = pm.mutate(off, p.INT_VARS, p.L_BOUND, p.U_BOUND)

    offspring = tr.truncate(np.vstack((elite, off)), p.INT_VARS)
    score, feasible = utils.evaluate(offspring, p.evaluate, p.MAX)

print(fittest)
print(f_best)
print(success)
