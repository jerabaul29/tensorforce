"""a class to help plotting dynamically. Adapted from:
    https://stackoverflow.com/questions/10944621/dynamically-updating-plot-in-matplotlib"""

import matplotlib.pyplot as plt

plt.ion()


class PltDynamicPlot():
    """dynamic plotting."""

    def __init__(self, min_x=0, max_x=10, min_y=-10, max_y=10, n_curves=1):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        
        self.n_curves = n_curves

        # Set up plot
        self.figure, self.ax = plt.subplots()
        
        self.dict_lines = {}
        
        for crrt_line_nbr in range(n_curves):
            self.dict_lines[crrt_line_nbr] = self.ax.plot([],[])[0]

        # Autoscale on unknown axis and known lims on the other
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)

    def update(self, xdata, ydata):
        # Update data (with the new _and_ the old points)
        for crrt_line_nbr in range(self.n_curves):
            crrt_line = self.dict_lines[crrt_line_nbr]
            crrt_line.set_xdata(xdata[crrt_line_nbr, :])
            crrt_line.set_ydata(ydata[crrt_line_nbr, :])

        # Need both of these in order to rescale
        # self.ax.relim()
        # self.ax.autoscale_view()
        # We need to draw *and* flush

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
