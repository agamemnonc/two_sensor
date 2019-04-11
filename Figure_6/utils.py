import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from seaborn.categorical import _CategoricalScatterPlotter
import warnings

# Subject numbers convention
# Able-bodied: 1-12
# Amputee: 20-21
SUBJECTS = np.concatenate((np.arange(1,13), np.array([20,21]))).astype(
        np.int32)

def get_df_early_late(df, early_limit=3, late_limit=7):
    """Creates a new data frame for storing results where trials are
    categorised as either ```early``` or ```late```.
    
    Parameters
    ----------
    df : DataFrame
        Data frame with results for each participant. 
    
    early_limit : int
        Limit for a trial to be considered as ```early``` (not inclusive).
    
    late_limit : int
        Limit for a trial to be considered as ```late``` (not inclusive).
    
    Returns
    -------
    df_block : DataFrame
        Data frame with results for early and late trials.
    """
    
    df_early_late = pd.DataFrame(columns=['Subject number', 'Subject type', 'Phase', 'Electrodes', 'Average EMG variance'])
    i = 0
    for subject in SUBJECTS:
        if subject == 20 or subject == 21:
            participant = "Amputee"
        else:
            participant = "Able-bodied"
        for electrodes in ['Used', 'Not used']:
            for phase, trials in zip(['Early', 'Late'], [[1,2], [9,10]]):
                avg_variance = df[
                    (df['Subject number']==subject) &
                    (df['Trial'].isin(trials)) &
                    (df['Electrodes']==electrodes)]['EMG variance'].mean()
                df_early_late.loc[i] = [subject, participant, phase,
                                 electrodes, avg_variance]
                i += 1
    
    return df_early_late


# Adapt the swarmplot a bit so that swarm points are sitting in the middle
# of boxplots
class _SwarmPlotter(_CategoricalScatterPlotter):
    def __init__(self, x, y, hue, data, order, hue_order,
                 dodge, orient, color, palette):
        """Initialize the plotter."""
        self.establish_variables(x, y, hue, data, orient, order, hue_order)
        self.establish_colors(color, palette, 1)

        # Set object attributes
        self.dodge = dodge
        self.width = .8

    def could_overlap(self, xy_i, swarm, d):
        """Return a list of all swarm points that could overlap with target.
        Assumes that swarm is a sorted list of all points below xy_i.
        """
        _, y_i = xy_i
        neighbors = []
        for xy_j in reversed(swarm):
            _, y_j = xy_j
            if (y_i - y_j) < d:
                neighbors.append(xy_j)
            else:
                break
        return np.array(list(reversed(neighbors)))

    def position_candidates(self, xy_i, neighbors, d):
        """Return a list of (x, y) coordinates that might be valid."""
        candidates = [xy_i]
        x_i, y_i = xy_i
        left_first = True
        for x_j, y_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(d ** 2 - dy ** 2) * 1.05
            cl, cr = (x_j - dx, y_i), (x_j + dx, y_i)
            if left_first:
                new_candidates = [cl, cr]
            else:
                new_candidates = [cr, cl]
            candidates.extend(new_candidates)
            left_first = not left_first
        return np.array(candidates)

    def first_non_overlapping_candidate(self, candidates, neighbors, d):
        """Remove candidates from the list if they overlap with the swarm."""

        # IF we have no neighbours, all candidates are good.
        if len(neighbors) == 0:
            return candidates[0]

        neighbors_x = neighbors[:, 0]
        neighbors_y = neighbors[:, 1]

        d_square = d ** 2

        for xy_i in candidates:
            x_i, y_i = xy_i

            dx = neighbors_x - x_i
            dy = neighbors_y - y_i

            sq_distances = np.power(dx, 2.0) + np.power(dy, 2.0)

            # good candidate does not overlap any of neighbors
            # which means that squared distance between candidate
            # and any of the neighbours has to be at least
            # square of the diameter
            good_candidate = np.all(sq_distances >= d_square)

            if good_candidate:
                return xy_i

        # If `position_candidates` works well
        # this should never happen
        raise Exception('No non-overlapping candidates found. '
                        'This should not happen.')

    def beeswarm(self, orig_xy, d):
        """Adjust x position of points to avoid overlaps."""
        # In this method, ``x`` is always the categorical axis
        # Center of the swarm, in point coordinates
        midline = orig_xy[0, 0]

        # Start the swarm with the first point
        swarm = [orig_xy[0]]

        # Loop over the remaining points
        for xy_i in orig_xy[1:]:

            # Find the points in the swarm that could possibly
            # overlap with the point we are currently placing
            neighbors = self.could_overlap(xy_i, swarm, d)

            # Find positions that would be valid individually
            # with respect to each of the swarm neighbors
            candidates = self.position_candidates(xy_i, neighbors, d)

            # Sort candidates by their centrality
            offsets = np.abs(candidates[:, 0] - midline)
            candidates = candidates[np.argsort(offsets)]

            # Find the first candidate that does not overlap any neighbours
            new_xy_i = self.first_non_overlapping_candidate(candidates,
                                                            neighbors, d)

            # Place it into the swarm
            swarm.append(new_xy_i)

        return np.array(swarm)

    def add_gutters(self, points, center, width):
        """Stop points from extending beyond their territory."""
        half_width = width / 2
        low_gutter = center - half_width
        off_low = points < low_gutter
        if off_low.any():
            points[off_low] = low_gutter
        high_gutter = center + half_width
        off_high = points > high_gutter
        if off_high.any():
            points[off_high] = high_gutter
        return points

    def swarm_points(self, ax, points, center, width, s, **kws):
        """Find new positions on the categorical axis for each point."""
        # Convert from point size (area) to diameter
        default_lw = mpl.rcParams["patch.linewidth"]
        lw = kws.get("linewidth", kws.get("lw", default_lw))
        dpi = ax.figure.dpi
        d = (np.sqrt(s) + lw) * (dpi / 72)

        # Transform the data coordinates to point coordinates.
        # We'll figure out the swarm positions in the latter
        # and then convert back to data coordinates and replot
        orig_xy = ax.transData.transform(points.get_offsets())

        # Order the variables so that x is the categorical axis
        if self.orient == "h":
            orig_xy = orig_xy[:, [1, 0]]

        # Do the beeswarm in point coordinates
        new_xy = self.beeswarm(orig_xy, d)

        # Transform the point coordinates back to data coordinates
        if self.orient == "h":
            new_xy = new_xy[:, [1, 0]]
        new_x, new_y = ax.transData.inverted().transform(new_xy).T

        # Add gutters
        if self.orient == "v":
            self.add_gutters(new_x, center, width)
        else:
            self.add_gutters(new_y, center, width)

        # Reposition the points so they do not overlap
        points.set_offsets(np.c_[new_x, new_y])

    def draw_swarmplot(self, ax, kws):
        """Plot the data."""
        s = kws.pop("s")

        centers = []
        swarms = []

        # Set the categorical axes limits here for the swarm math
        if self.orient == "v":
            ax.set_xlim(-.5, len(self.plot_data) - .5)
        else:
            ax.set_ylim(-.5, len(self.plot_data) - .5)

        # Plot each swarm
        for i, group_data in enumerate(self.plot_data):

            if self.plot_hues is None or not self.dodge:

                width = self.width

                if self.hue_names is None:
                    hue_mask = np.ones(group_data.size, np.bool)
                else:
                    hue_mask = np.array([h in self.hue_names
                                         for h in self.plot_hues[i]], np.bool)
                    # Broken on older numpys
                    # hue_mask = np.in1d(self.plot_hues[i], self.hue_names)

                swarm_data = group_data[hue_mask]

                # Sort the points for the beeswarm algorithm
                sorter = np.argsort(swarm_data)
                swarm_data = swarm_data[sorter]
                point_colors = self.point_colors[i][hue_mask][sorter]

                # Plot the points in centered positions
                cat_pos = np.ones(swarm_data.size) * i
                kws.update(c=point_colors)
                if self.orient == "v":
                    points = ax.scatter(cat_pos, swarm_data, s=s, **kws)
                else:
                    points = ax.scatter(swarm_data, cat_pos, s=s, **kws)

                centers.append(i)
                swarms.append(points)

            else:
                offsets = self.hue_offsets/2
                width = self.nested_width

                for j, hue_level in enumerate(self.hue_names):
                    #if i == 0 && j == 0:
                     #   offsets = offsets + 0.05
                    hue_mask = self.plot_hues[i] == hue_level
                    swarm_data = group_data[hue_mask]

                    # Sort the points for the beeswarm algorithm
                    sorter = np.argsort(swarm_data)
                    swarm_data = swarm_data[sorter]
                    point_colors = self.point_colors[i][hue_mask][sorter]

                    # Plot the points in centered positions
                    center = i + offsets[j]
                    cat_pos = np.ones(swarm_data.size) * center
                    kws.update(c=point_colors)
                    if self.orient == "v":
                        points = ax.scatter(cat_pos, swarm_data, s=s, **kws)
                    else:
                        points = ax.scatter(swarm_data, cat_pos, s=s, **kws)

                    centers.append(center)
                    swarms.append(points)

        # Update the position of each point on the categorical axis
        # Do this after plotting so that the numerical axis limits are correct
        for center, swarm in zip(centers, swarms):
            if swarm.get_offsets().size:
                self.swarm_points(ax, swarm, center, width, s, **kws)

    def plot(self, ax, kws):
        """Make the full plot."""
        self.draw_swarmplot(ax, kws)
        self.add_legend_data(ax)
        self.annotate_axes(ax)
        if self.orient == "h":
            ax.invert_yaxis()

def swarmplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
              dodge=False, orient=None, color=None, palette=None,
              size=5, edgecolor="gray", linewidth=0, ax=None, **kwargs):

    if "split" in kwargs:
        dodge = kwargs.pop("split")
        msg = "The `split` parameter has been renamed to `dodge`."
        warnings.warn(msg, UserWarning)

    plotter = _SwarmPlotter(x, y, hue, data, order, hue_order,
                            dodge, orient, color, palette)
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("zorder", 3)
    size = kwargs.get("s", size)
    if linewidth is None:
        linewidth = size / 10
    if edgecolor == "gray":
        edgecolor = plotter.gray
    kwargs.update(dict(s=size ** 2,
                       edgecolor=edgecolor,
                       linewidth=linewidth))

    plotter.plot(ax, kwargs)
    return ax
