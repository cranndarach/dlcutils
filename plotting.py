#!/usr/bin/env python3

"""
Utility functions for plotting.
"""

import matplotlib.pyplot as plt


def get_trajectory(fnc, x0, steps):
    """
    First, a function to calculate the trajectory. Takes a function,
    a starting x-value, and a number of steps. Returns a pair of lists:
    (time, trajectory).
    """
    step = 0
    xx = x0
    time = list(range(steps+1))
    trajectory = [x0]
    while step < steps:
        step += 1
        xx = fnc(xx)
        trajectory.append(xx)
    return (time, trajectory)


def plot_curve(*args, **kwargs):
    """Given x and y values, plot a curve."""
    # Label the axes according to the arguments, or use defaults.
    plot_kwargs = {}
    x_lab = kwargs.get("x_label", "x")
    y_lab = kwargs.get("y_label", "y")
    label = kwargs.get("label", None)
    title = kwargs.get("title", "")
    axes = kwargs.get("axes", False)
    # fmt = kwargs.get("fmt", None)
    axes_fmt = kwargs.get("axes_fmt", "w-")
    # If user specifies a line format, prep it to be passed
    # to the plot fnc.
    # if fmt:
    #     plot_args = (fmt)
    # else:
    #     plot_args = ()
    # If user specified a label (for the legend), add it to plot_kwargs.
    if label:
        plot_kwargs["label"] = label

    # Plot the x and y values, pass the arguments, if any.
    plt.plot(*args, **plot_kwargs)

    # If told to plot axes, pass the plot values to the
    # plot_axes fnc, with the format string (if specified).
    if axes:
        plot_axes(args[0], args[1], fmt=axes_fmt)

    # Use specified or default titles/labels, determined above.
    plt.title(title, fontsize=16)
    plt.xlabel(x_lab, fontsize=16)
    plt.ylabel(y_lab, fontsize=16)

    if label:
        plt.legend()


def plot_axes(xs, ys, **kwargs):
    """Plot x and y axes, given x and y values."""
    fmt = kwargs.get("fmt", "w-")

    plt.plot([min(xs), max(xs)], [0, 0], fmt)
    plt.plot([0, 0], [min(ys), max(ys)], fmt)


def plot_cobweb(fnc, x0, steps, **kwargs):
    """
    Given a function, a starting value, and a number of steps,
    make a cobweb plot.
    """
    t = 0
    xt = x0
    points = []
    yx = [(0, 0)]

    starting_fx = min([0, fnc(x0)])
    fx = [(starting_fx, fnc(starting_fx))]
    while t < steps:
        # First add the point on y = x.
        points.append((xt, xt))
        yx.append((xt, xt))
        # Then incrememt t and find next value on y = f(x).
        t += 1
        xlast = xt
        xt = fnc(xlast)
        points.append((xlast, xt))
        fx.append((xlast, xt))
    xs, ys = zip(*points)
    fx_x, fx_y = zip(*fx)
    yx_x, yx_y = zip(*yx)

    if kwargs.get("fmt", False):
        plot_args = (xs, ys, kwargs["fmt"])
    else:
        plot_args = (xs, ys)

    kwargs["x_label"] = kwargs.get("x_label", "x(t)")
    kwargs["y_label"] = kwargs.get("y_label", "x(t+1)")

    plt.plot(fx_x, fx_y)
    plt.plot(yx_x, yx_y)
    plot_curve(*plot_args, **kwargs)
    # Plot the x-axis.
    xmin = min(list(xs)+[0])
    xmax = max(list(xs)+[0])
    ymin = min(list(ys)+[0])
    ymax = max(list(ys)+[0])

    plt.plot([xmin, xmax], [0, 0], "w-")
    # Plot the y-axis.
    plt.plot([0, 0], [ymin, ymax], "w-")


def plot_trajectory(fnc, x0, steps, **kwargs):
    """
    Function to plot a trajectory given a function, a starting x-value,
    a number of steps, and keyword arguments to be passed to the plotting
    function.
    """
    # Using the function from just above.
    xs, ys = get_trajectory(fnc, x0, steps)
    if kwargs.get("fmt", False):
        plot_args = (xs, ys, kwargs["fmt"])
    else:
        plot_args = (xs, ys)
    # And a function from the previous assignment, now saved in a library.
    plot_curve(*plot_args, **kwargs)


def state_plot(mtx, ax, color_0="#2B91E0", color_1="#5F35E4"):
    """
    Takes a matrix of 0s and 1s, and a color corresponding to 0 and to
    1. Plots a rectangle of the given color at each 1 and at each 0.
    All rows in the matrix must be of equal length. Like
    two_color_plot() but but needs to be passed an Axes object.
    """
    # Make points square.
    ax.set_aspect("equal")
    # Make it go from top to bottom.
    mtx.reverse()

    # Set the size of the plot based on the length of the list (number
    # of rows) and the length of the first item in the list (number of
    # columns).
    width = len(mtx[0])
    height = len(mtx)
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    # Get iterator of all possible points by calculating the cartesian
    # product of the indices of each dimension.
    points_iter = itertools.product(range(width), range(height))

    for point in points_iter:
        col, row = point
        if mtx[row][col] == 0:
            ax.add_patch(make_patch(point, color_0))
        elif mtx[row][col] == 1:
            ax.add_patch(make_patch(point, color_1))
        else:
            raise ValueError("Unrecognized value " + str(mtx[row][col]) +
                             ". Input matrix should only contain 1s and 0s.")
