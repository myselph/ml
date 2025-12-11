import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

import matplotlib
matplotlib.use('TkAgg')


class CartPoleVisualizer:
    # Constants for visualization
    CART_WIDTH = 0.4
    CART_HEIGHT = 0.2
    POLE_WIDTH = 0.03
    POLE_LENGTH = 1.0
    AXLE_RADIUS = 0.05
    PLOT_YLIM = [-0.5, POLE_LENGTH + CART_HEIGHT + 0.5]
    # Track length is a parameter, not a constant, so it can be used by the
    # calling code to determine boundary violations.

    def __init__(self, track_length: float):
        plt.ion()
        self.plot_xlim = [-CartPoleVisualizer.CART_WIDTH / 2 - track_length / 2,
                          CartPoleVisualizer.CART_WIDTH / 2 + track_length / 2]
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_xlim(self.plot_xlim)
        self.ax.set_ylim(CartPoleVisualizer.PLOT_YLIM)
        self.ax.set_aspect('equal')
        self.ax.set_title('Cart-Pole Simulation')
        self.ax.set_xlabel('Position (m)')
        self.ax.set_ylabel('Height (m)')
        self.ax.grid(True)

        # Initialize drawing elements (artists)
        self.cart = patches.Rectangle(
            (0, 0), CartPoleVisualizer.CART_WIDTH, CartPoleVisualizer.CART_HEIGHT, fc='blue')
        self.ax.add_patch(self.cart)

        self.pole = patches.Rectangle(
            (0, 0), CartPoleVisualizer.POLE_WIDTH, CartPoleVisualizer.POLE_LENGTH, fc='red')
        self.ax.add_patch(self.pole)

        self.axle = patches.Circle(
            (0, 0), CartPoleVisualizer.AXLE_RADIUS, fc='black')
        self.ax.add_patch(self.axle)

        # Track
        self.track = self.ax.hlines(
            0, self.plot_xlim[0], self.plot_xlim[1], colors='gray', linestyles='solid', linewidth=2)

        # Initial display of the figure
        # display(self.fig)
        plt.pause(0.001)

    def update_display(self, cart_position: float, pole_angle: float):

        # Cart position
        cart_x = cart_position - CartPoleVisualizer.CART_WIDTH / 2
        cart_y = 0  # Cart sits on the track
        self.cart.set_xy((cart_x, cart_y))

        # Pole position and angle
        # Axle is at the top center of the cart
        axle_x = cart_position
        axle_y = cart_y + CartPoleVisualizer.CART_HEIGHT
        self.axle.set_center((axle_x, axle_y))

        # Set the bottom-left corner of the pole rectangle
        pole_x = axle_x - CartPoleVisualizer.POLE_WIDTH / 2
        pole_y = axle_y
        self.pole.set_xy((pole_x, pole_y))

        # Create a transform to rotate the pole around the axle point
        pole_transform = plt.matplotlib.transforms.Affine2D() \
            .translate(-axle_x, -axle_y) \
            .rotate(-pole_angle) \
            .translate(axle_x, axle_y) + self.ax.transData
        self.pole.set_transform(pole_transform)

        plt.pause(0.001)


if __name__ == '__main__':
    visualizer = CartPoleVisualizer(track_length=5)
    # Simulate a few steps
    cart_positions = np.linspace(-2, 2, 50)
    pole_angles = np.linspace(np.pi/4, -np.pi/4, 50)  # -45 to +45 degrees

    for i in range(50):
        visualizer.update_display(cart_positions[i], pole_angles[i])

    print("Simulation example finished.")
