#!/usr/bin/env python3

"""
Cellular automaton functions and classes.
"""

import sys
import random as rd
import numpy as np
import matplotlib.pyplot as plt


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
            state_plot(list(state_list), cell_axes)
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


