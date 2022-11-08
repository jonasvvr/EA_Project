import plots
from r0123456 import r0123456

ea = r0123456()

def plot_lambda():
    values = [50, 100, 150, 200]

    g = plots.Graph('lambda_graph')
    for val in values:
        ea.optimize('tour50.csv', lam=val)
        obj_value = ea.reporter.bestObjectiveList
        iters = ea.reporter.iterationsList
        g.addSeries(f'lambda={val}', iters, obj_value)

    g.commit(x_label='iterations', y_label='objective_value')

def plot_to_mutate():
    values = [1, 3, 5, 7]

    g1 = plots.Graph('to_mutate_graph')
    for val in values:
        ea.optimize('tour50.csv', to_mutate=val)
        obj_value = ea.reporter.bestObjectiveList
        iters = ea.reporter.iterationsList
        g1.addSeries(f'to_mutate={val}', iters, obj_value)

    g1.commit(x_label='iterations', y_label='objective_value')


def plot_mutation_tries():
    values = [10, 20, 30, 40]

    g2 = plots.Graph('mutation_tries_graph')
    for val in values:
        ea.optimize('tour50.csv', mutation_tries=val)
        obj_value = ea.reporter.bestObjectiveList
        iters = ea.reporter.iterationsList
        g2.addSeries(f'mutation_tries={val}', iters, obj_value)

    g2.commit(x_label='iterations', y_label='objective_value')


def plot_k():
    values = [1, 3, 5, 10]

    g3 = plots.Graph('k_graph')
    for val in values:
        ea.optimize('tour50.csv', k=val)
        obj_value = ea.reporter.bestObjectiveList
        iters = ea.reporter.iterationsList
        g3.addSeries(f'k={val}', iters, obj_value)

    g3.commit(x_label='iterations', y_label='objective_value')


def plot_alpha():
    values = [0.025, 0.05, 0.075, 0.10]

    g4 = plots.Graph('alpha_graph')
    for val in values:
        ea.optimize('tour50.csv', alph=val)
        obj_value = ea.reporter.bestObjectiveList
        iters = ea.reporter.iterationsList
        g4.addSeries(f'alpha={val}', iters, obj_value)

    g4.commit(x_label='iterations', y_label='objective_value')


plot_lambda()
plot_alpha()
plot_k()
plot_mutation_tries()
plot_to_mutate()