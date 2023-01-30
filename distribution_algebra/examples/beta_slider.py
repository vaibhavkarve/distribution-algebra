#!/usr/bin/env python3

from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from distribution_algebra.beta import Beta

matplotlib.rcParams['font.size'] = 14

LINX = np.linspace(0, 1, 1000)
INIT_ALPHA = 1
INIT_BETA = 1

# Create the figure and the line that we will manipulate
fig: plt.figure.Figure  # type: ignore
ax: plt.axes.Axes  # type: ignore
fig, ax = plt.subplots()
line, = ax.plot(LINX, Beta(alpha=INIT_ALPHA, beta=INIT_BETA).pdf(LINX), lw=3)
ax.set_ylim(0, 4)
ax.set_xlabel('x')
ax.set_ylabel('Probability density')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.1, bottom=0.4)

# Make a horizontal slider to control the frequency.
axhorizontal1 = fig.add_axes([0.2, 0.2, 0.65, 0.03])
freq_slider = Slider(
    ax=axhorizontal1,
    label='Alpha = 1 + Failures',
    valmin=0,
    valmax=10,
    valinit=INIT_ALPHA,
    valstep=1,
    color="red",
)

# Make a vertically oriented slider to control the amplitude
axhorizontal2 = fig.add_axes([0.2, 0.1, 0.65, 0.03])
amp_slider = Slider(
    ax=axhorizontal2,
    label="Beta = 1 + Successes",
    valmin=0,
    valmax=10,
    valstep=1,
    valinit=INIT_BETA,
    color="green",
)


# The function to be called anytime a slider's value changes
def update(_: None) -> None:
    line.set_ydata(Beta(alpha=cast(float, amp_slider.val),  # type: ignore
                        beta=cast(float, freq_slider.val)).pdf(LINX))  # type: ignore
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(_: None) -> None:
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()
