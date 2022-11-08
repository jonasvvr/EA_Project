import json
from abc import ABC, abstractmethod
from pathlib import Path

from matplotlib import pyplot as plt

ASPECT_RATIO_SIZEUP = 3
PATH_FIGURES = Path(__file__).resolve().parent / "plots"
PATH_FIGURES.mkdir(exist_ok=True)
PATH_RAW = Path(__file__).resolve().parent / "raw"
PATH_RAW.mkdir(exist_ok=True)

from matplotlib import colors as mcolors

MPL_COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
COLORS = [
    MPL_COLORS.get("r"), MPL_COLORS.get("g"), MPL_COLORS.get("b"),
    MPL_COLORS.get("lime"), MPL_COLORS.get("darkviolet"), MPL_COLORS.get("gold"),
    MPL_COLORS.get("cyan"), MPL_COLORS.get("magenta"), MPL_COLORS.get("firebrick")
]


def getColours():
    return list(COLORS)


class Diagram(ABC):
    """
        Superclass for Graph class
        Responsible for Writing Diagrams
    """

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def safeFigureWrite(stem: str, suffix: str, figure):
        print(f"Writing figure {stem} ...")
        modifier = 0
        while (PATH_FIGURES / (stem + "_" + str(modifier) + suffix)).exists():
            modifier += 1
        figure.savefig(PATH_FIGURES.as_posix() + "/" + stem + "_" + str(modifier) + suffix, bbox_inches='tight')

    @staticmethod
    def safeDatapointWrite(stem: str, data: dict):
        print(f"Writing json {stem} ...")
        modifier = 0
        while (PATH_RAW / (stem + "_" + str(modifier) + ".json")).exists():
            modifier += 1
        with open(PATH_RAW / (stem + "_" + str(modifier) + ".json"), "w") as file:
            json.dump(data, file)


class Graph(Diagram):

    def __init__(self, name: str):
        super().__init__(name)
        self.series = dict()

    def add_point(self, name: str, x, y):
        series = self.series.get(name)
        if series is None:
            self.series[name] = ([], [])
        self.series[name][0].extend(x)
        self.series[name][1].extend(y)

    def addSeries(self, name: str, xs: list, ys: list):
        if name not in self.series:
            self.series[name] = ([], [])
        self.series[name][0].extend(xs)
        self.series[name][1].extend(ys)

    def commit(self, aspect_ratio=(4, 3), x_label="", y_label="",
               do_points=True, save_dont_display=True,
               grid_linewidth=1, curve_linewidth=1,
               x_lims=None, y_lims=None,
               fig: plt.Figure = None, main_ax: plt.Axes = None):
        # Figure stuff
        # styles = {"r.-", "g.-"}
        colours = getColours()
        if fig is None or main_ax is None:
            fig, main_ax = plt.subplots(
                figsize=(ASPECT_RATIO_SIZEUP * aspect_ratio[0], ASPECT_RATIO_SIZEUP * aspect_ratio[1]))
        main_ax.grid(True, which='both', linewidth=grid_linewidth)
        main_ax.axhline(y=0, color='k', lw=0.5)

        style = ".-" if do_points else "-"

        for name, samples in self.series.items():
            main_ax.plot(samples[0], samples[1], style, c=colours.pop(0), label=name, linewidth=curve_linewidth)

        if x_label:
            main_ax.set_xlabel(x_label)
        if y_label:
            main_ax.set_ylabel(y_label)
        main_ax.legend(loc='upper right')

        if x_lims:
            main_ax.set_xlim(x_lims[0], x_lims[1])
        if y_lims:
            main_ax.set_ylim(y_lims[0], y_lims[1])

        # File stuff
        if save_dont_display:
            Diagram.safeDatapointWrite(self.name, self.series)
            Diagram.safeFigureWrite(self.name, ".pdf", fig)
        else:
            show_figure(fig)


def show_figure(fig: plt.Figure):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.show()
