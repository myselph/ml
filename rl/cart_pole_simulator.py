from cart_pole_visualizer import CartPoleVisualizer
import numpy as np
import time


class BasicCartPole:
    TRACK_LENGTH = 5
    MASS_CART = 1  # kg
    MASS_STICK = 0.1
    G = 9.81  # m/s^2
    VIS_UPDATE_INTERVAL = 0.05

    def __init__(self, x, v, angle, visualize, dt):
        self.x = x
        self.v = v
        self.angle = angle
        self.rot_v = np.array([0.0]*len(x), dtype=np.float32)
        self.dt = dt
        self.visualize = visualize
        self.t_sim = 0
        self.ok_state = np.array([True]*len(x))
        if visualize:
            assert len(
                x) == 1, "Visualization only supported for scalar input arguments"
            self.visualizer = CartPoleVisualizer(self.TRACK_LENGTH)
            self.t_real_start = time.time()
            self.t_vis_update = 0

    def apply_and_step(self, f):
        self.last_f = f
        self.t_sim += self.dt

    def maybe_plot(self):
        if not self.visualize:
            return
        t_real = time.time()
        # Throttle plotting so we stay in real time.
        if t_real - self.t_vis_update > BasicCartPole.VIS_UPDATE_INTERVAL:
            self.t_vis_update = t_real
            self.visualizer.update_display(self.x[0], self.angle[0])
        format_string = (
            f"t_real={t_real-self.t_real_start:.1f}, t_sim={self.t_sim:.1f}, x={self.x[0]:.2f}, "
            f"v={self.v[0]:.2f}, angle={180/np.pi*self.angle[0]:.1f}, rot_v={self.rot_v[0]:.2f}, f={self.last_f[0]:.1f}"
        )
        print(format_string)
        t_missing = self.t_sim - (time.time()-self.t_real_start)
        if t_missing > 0:
            time.sleep(t_missing)

    def ok(self, quiet=False):
        """
        Returns a boolean numpy array indicating which simulations are ongoing and which ones have failed.
        """
        self.ok_state = np.logical_and.reduce((
            self.ok_state, self.angle <= np.pi/2, self.angle >= -
            np.pi/2, self.x >= -self.TRACK_LENGTH / 2,
            self.x <= self.TRACK_LENGTH / 2))
        return self.ok_state

    def get_state(self):
        """
        Returns the internal state of the simulator as a B x 4 array
        """
        return np.stack([self.x, self.v, self.angle, self.rot_v], axis=1)


class SimpleCartPole(BasicCartPole):
    """
    A simple, approximate pole-on-a-cart simulation that I wrote on a flight.
    It takes care of the physics, and also plots a visualization.
    Assumptions:
    1. The cart movement follows F = m_cart * a_cart and we assume the stick is so
        lightweight it can be ignored for the cart's movement
    2. The stick, attached to the cart with a rotary joint, experiences two forces
       that lead to angular acceleration via tangential forces:
       a) gravity
       b) stick_acceleration == cart acceleration -> F = m_stick * a_cart
    3. No friction.
    That's pretty coarse but it looks good enough and allows us to train a policy.
    """

    def __init__(self, x=np.array([0.0], dtype=np.float32), v=np.array([0.0],
                 dtype=np.float32), angle=np.array([0.0], dtype=np.float32), visualize=False, dt=0.02):
        super().__init__(x, v, angle, visualize, dt)
        self.stick_gravity = self.MASS_STICK * self.G

    def apply_and_step(self, force: np.array):
        super().apply_and_step(force)
        # Apply force to car and integrate new velocity + position.
        accel = force/self.MASS_CART
        self.v = self.v + self.dt * accel
        self.x = self.x + self.dt * self.v

        # Calculate new stick angle and angular velocity.
        # Force due to gravity

        f_gravity_tangential = 0.5 * np.sin(self.angle) * self.stick_gravity
        # Force due to car acceleration of cart.
        f_cart_horizontal = - self.MASS_STICK * accel / 2
        f_cart_tangential = np.cos(self.angle) * f_cart_horizontal
        f_tangential = f_gravity_tangential + f_cart_tangential

        self.rot_v = self.rot_v + self.dt * f_tangential / self.MASS_STICK
        self.angle = self.angle + self.dt * self.rot_v


class RealisticCartPole(BasicCartPole):
    """
    More realistic equations from https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
    """
    MU_C = 0.01  # friction coefficient between cart and track
    MU_P = 0.001  # friction coefficient between pole and cart
    POLE_LENGTH = 1

    def __init__(self, x=np.array([0.0], dtype=np.float32), v=np.array([0.0],
                 dtype=np.float32), angle=np.array([0.0], dtype=np.float32), visualize=False, dt=0.02):
        super().__init__(x, v, angle, visualize, dt)

    def apply_and_step(self, force):
        super().apply_and_step(force)
        cart_accel = RealisticCartPole.MASS_STICK * \
            RealisticCartPole.G * np.sin(self.angle)*np.cos(self.angle)
        cart_accel -= 7/3*(force+RealisticCartPole.MASS_STICK*RealisticCartPole.POLE_LENGTH /
                           2 * self.rot_v*self.rot_v*np.sin(self.angle) - RealisticCartPole.MU_C*self.v)
        cart_accel -= RealisticCartPole.MU_P * self.rot_v * \
            np.cos(self.angle)*2/RealisticCartPole.POLE_LENGTH
        cart_accel /= (RealisticCartPole.MASS_STICK*np.cos(self.angle)
                       * np.cos(self.angle)-7/3*(self.MASS_STICK+self.MASS_CART))

        rot_accel = 3/7*2/RealisticCartPole.POLE_LENGTH*(RealisticCartPole.G*np.sin(self.angle)-cart_accel*np.cos(
            self.angle)-RealisticCartPole.MU_P*self.rot_v/self.MASS_STICK*2/RealisticCartPole.POLE_LENGTH)

        self.v = self.v + self.dt * cart_accel
        self.x = self.x + self.dt * self.v
        self.rot_v = self.rot_v + self.dt * rot_accel
        self.angle = self.angle + self.dt * self.rot_v


def simple_strategy():
    """
    Try to balance the stick by applying a force proportional to the angle.
    """
    cartpole = RealisticCartPole(x=np.array([0.0]), v=np.array([0.0]),
                               angle=np.array([np.pi/8]), visualize=True)
    for t in np.arange(0, 10, cartpole.dt):
        f = 180.0 / np.pi * cartpole.angle / 0.5
        cartpole.apply_and_step(f)
        cartpole.maybe_plot()
        if not cartpole.ok()[0]:
            print("Fail!")
            break


if __name__ == '__main__':
    simple_strategy()
