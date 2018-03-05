#!/usr/bin/env python3

"""
Cellular automaton functions and classes.
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Voter:
    def __init__(self, *args, **kwargs):
        if kwargs.get("data", False):
            self.data = kwargs["data"]
        else:
            self.initial, self.max_steps, self.adj_mat = args
            self.length = len(self.initial)
            self.sequence = range(self.length)

    def get_pct_on(self, row):
        """
        Calculates the percent of the state that is currently on 1.
        """
        return round((sum(row) / len(row))*100)

    def get_neighbors(self, index):
        """
        Gets the neighbors of a particular cell.
        """
        roi = self.adj_mat[index]
        neighbors = []
        for pos in range(len(roi)):
            if roi[pos] == 1:
                neighbors.append(pos)
        return neighbors

    def get_all_neighbors(self):
        """
        Returns a list of neighbors of each cell.
        """
        self.all_neighbors = [self.get_neighbors(pos) for pos in self.sequence]

    def next_cell_state(self, state, pos):
        """
        Gets the next state of a single cell by randomly selecting from
        the "on-ness" of that cell's neighbors.
        """
        return rd.choice([state[nbr] for nbr in self.all_neighbors[pos]])

    def voter_generator(self):
        """
        Generator function that yields the state, percent of 1s, and
        step number at each step in a run.
        """
        if not hasattr(self, "all_neighbors"):
            self.get_all_neighbors()
        step = 0
        state = self.initial.copy()
        while step <= self.max_steps:
            yield (state, self.get_pct_on(state), step)
            if len(set(state)) == 1:
                break
            step += 1
            state = [self.next_cell_state(state, pos) for pos in self.sequence]

    def run_sim(self, runs):
        """
        Runs the voter simulation the specified number of times, returns a
        list of results.
        """
        self.data = [[step for step in self.voter_generator()] for _ in
                     range(runs)]

    def get_stats(self, quiet=False):
        """
        Neatens up the data and calculates the mean and SD of the steps
        per run (and maybe eventually some other stats).
        """
        self.reshaped = [list(zip(*run)) for run in self.data]
        self.states, self.progress, self.steps = list(zip(*self.reshaped))
        self.steps_list = [max(step_seq) for step_seq in self.steps]
        self.mean_steps = np.mean(self.steps_list)
        self.sd_steps = np.std(self.steps_list)
        if not quiet:
            self.print_stats()

    def steps_to_converge(self):
        gen = self.voter_generator()
        while True:
            try:
                state, pct_on, step = next(gen)
            except StopIteration:
                break
        return step

    def print_stats(self):
        dlc.print_mixed("Mean number of steps:", np.mean(self.steps_list))
        dlc.print_mixed("Standard deviation:", np.std(self.steps_list))

    def plot_percent(self, show=False, fig=1):
        plt.close("all")
        plt.figure(fig)
        for prog in self.progress:
            xs = range(len(prog))
            dlc.plot_curve(xs, prog)
        dlc.plot_axes([0, max(self.steps_list)], [0, 100])
        plt.title("Percentage of 1s over time")
        plt.xlabel("step")
        plt.ylabel("percentage of 1s")
        if show:
            plt.show()

    def plot_steps(self, **kwargs):
        plt.close("all")
        plt.figure(kwargs.get("fig", 1))
        density = stats.gaussian_kde(self.steps_list)
        xs = np.linspace(0, max(self.steps_list), 200)
        if not kwargs.get("title", False):
            kwargs["title"] = "Density of steps to convergence"
        dlc.plot_curve(xs, density(xs), **kwargs)
        if kwargs.get("show", False):
            plt.show()

    def plot_states(self, **kwargs):
        plt.close("all")
        # Make a grid of 3 columns and however many rows are needed.
        rows = int(len(self.states)/3)
        # If there's a remainder, add an extra row.
        if (len(self.states) % 3):
            rows += 1
        fig, axes = plt.subplots(rows, 3, figsize=(20, 20))
        plt_no = 0
        for state_list in self.states:
            row = plt_no % 3
            col = int(plt_no/3)
            cell_axes = axes[col][row]
            dlc.state_plot(list(state_list), cell_axes)
            plt_no += 1
        if kwargs.get("show", False):
            plt.show()

    def proportion_positive_convergence(self):
        # states[-1] is the last state in the list. set() returns the
        # set of unique items. because there should only be one item
        # in the set if it converged, sum() will just return that item.
        end_states = [sum(set(states[-1])) for states in self.states]
        return (sum(end_states) / len(end_states))


def random_state(length):
    """
    Return a list of specified length with each element randomly set to
    either 0 or 1.
    """
    return [rd.randint(0, 1) for _ in range(length)]


def get_cell_neighborhood(seq, pos):
    """
    Return the neighborhood (a triple) of the cell in the specified
    position of a sequence.
    """
    cell = seq[pos]
    left = seq[-1] if pos == 0 else seq[pos-1]
    right = seq[0] if pos == (len(seq) - 1) else seq[pos+1]
    return (left, cell, right)


def get_wolfram_rule(rule_no):
    """
    Return a dictionary with the cell => next_state correspondences
    for a rule number for a Wolfram CA.
    """
    cell_types = [
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (1, 0, 0),
        (0, 1, 1),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, 0)
    ]
    rule_bin = str(bin(rule_no))
    rule_substr = rule_bin[2::]
    padding = "0" * (8 - len(rule_substr))
    rule_string = padding + rule_substr
    rule_seq = [int(num) for num in rule_string]
    rule_corresp = dict(zip(cell_types, rule_seq))
    return rule_corresp


def wolfram_cell_step_from_neighborhood(neighborhood, rule):
    """
    Simple utility fnc to get the result of a rule for a neighborhood.
    Exists to make list comprehensions comprehensible.
    """
    return rule[neighborhood]


def wolfram_cell_step(seq, pos, rule):
    """
    Simple utility fnc to get the next state of a cell in the specified
    position of the sequence. Exists for fnc composition/application.
    """
    nbrs = get_cell_neighborhood(seq, pos)
    return wolfram_cell_step_from_neighborhood(nbrs, rule)


def wolfram_next_step(initial, rule):
    """
    Given the initial state of a CA and a rule correspondence (see
    `get_wolfram_rule()`), return the next state.
    """
    return [wolfram_cell_step(initial, pos, rule) for pos in
            range(len(initial))]


def wolfram_plot(initial, rule_no, steps, color0="#2B91E0", color1="#5F35E4"):
    """
    Plots the states of a Wolfram CA starting with the `initial`
    state, using rule `rule_no`, and running for `steps` steps.
    """
    states = wolfram_steps(initial, rule_no, steps)
    states.reverse()
    two_color_plot(states, color0, color1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Rule #" + str(rule_no))


def wolfram_steps(initial, rule_no, steps):
    """
    Returns list of `steps` steps of a Wolfram CA given the `initial`
    state and a rule number.
    """
    rule = get_wolfram_rule(rule_no)
    step = 0
    state = initial.copy()
    states = [state]
    while step < steps:
        step += 1
        state = wolfram_next_step(state, rule)
        states.append(state)
    return states
