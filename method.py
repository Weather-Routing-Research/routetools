import time

import cma
import numpy as np


def optimize(
    u,
    v,
    src,
    dst,
    travel_speed=None,
    travel_time=None,
    K=6,
    L=64,
    popsize=200,
    tolfun=1e-4,
    verbose=True,
    **kwargs,
):
    """
    Solve the vessel routing problem for a given vector field. Two modes are supported:
        - Fixed speed-through-water. Optimize the vessel heading
        - Fixed total travel time. Optimize heading and speed-through-water

    Algorithm: parameterize the space of solutions with a Bézier curve, and optimize the control points using the CMA-ES optimization method.

    Parameters
    ----------
    u : callable
        Horizontal component of the field. Accepts (x, y)
    v : callable
        Vertical component of the field. Accepts (x, y)
    src : np.ndarray
        Source point (2D)
    dst : np.ndarray
        Destination point (2D)
    travel_speed : float, optional
        The boat will have this fixed speed w.r.t. the water. If set, then `travel_time` must be None. By default None
    travel_time : float, optional
        The boat can regulate its speed but must complete the path in exactly this time. If set, then `travel_speed` must be None
    K : int, optional
        Number of free Bézier control points. By default 6
    L : int, optional
        Number of points evaluated in each Bézier curve. By default 64
    popsize : int, optional
        Population size for the CMA-ES optimizer. By default 200
    tolfun : float, optional
        Tolerance for the optimizer. By default 1e-4
    verbose : bool, optional
        By default True
    **kwargs: optional additional parameters for CMA-ES

    Returns
    -------

    np.ndarray
        The optimized curve (shape L x 2)
    """

    def control_to_curve(control):
        """
        Convert a batch of free parameters into a batch of Bézier curves
        :param x: a B x 2K matrix. The first K columns are the x positions of the Bézier control points, and the last K are the y positions
        :return: curves, control points (both batched)
        """

        def batch_bezier(t, control):
            """
            Evaluate a batch of Bézier curves (using de Casteljau's algorithm)

            :param t: evaluation points (vector of shape K), all between 0 and 1
            :param control: batched matrix of control points, with shape B x P x N
            :return: batch of curves (matrix of shape B x K x N)
            """

            control = np.tile(control[:, :, None, :], [1, 1, len(t), 1])
            while control.shape[1] > 1:
                control = (1 - t[None, None, :, None]) * control[:, :-1, :, :] + t[
                    None, None, :, None
                ] * control[:, 1:, :, :]
            return control[:, 0, ...]

        control = control.reshape(control.shape[0], -1, 2)

        # Add the fixed endpoints
        first_point = np.broadcast_to(src, (control.shape[0], 1, 2))
        last_point = np.broadcast_to(dst, (control.shape[0], 1, 2))
        control = np.hstack([first_point, control, last_point])
        return batch_bezier(t=np.linspace(0, 1, L), control=control)

    def cost_function(curve):
        """
        Compute the fuel consumption of a batch of paths navigating over a vector field. Two modes are supported:
            - Fixed travel speed (w.r.t. water)
            - Fixed travel time

        :param u: horizontal component of the field (callable)
        :param v: vertical component of the field (callable)
        :param curve: batch of trajectories (an array of shape B x L x 2)
        :param travel_speed: the boat will have this fixed speed w.r.t. the water. If set, then `travel_time` must be None
        :param travel_time: the boat can regulate its speed but must complete each path in exactly this time. If set, then `travel_speed` must be None
        :return: a batch of scalars (vector of shape B)
        """

        curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
        curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
        uinterp = u(curvex, curvey)
        vinterp = v(curvex, curvey)

        # Curve segments
        delta_x = np.diff(curve[:, :, 0], axis=1)
        delta_y = np.diff(curve[:, :, 1], axis=1)

        if travel_time is None:  # We navigate the path at fixed speed
            # Source: Zermelo's problem for constant wind
            # https://en.wikipedia.org/wiki/Zermelo%27s_navigation_problem#Constant-wind_case
            v2 = travel_speed**2
            w2 = uinterp**2 + vinterp**2
            dw = delta_x * uinterp + delta_y * vinterp
            d2 = delta_x**2 + delta_y**2  # Segment lengths
            cost = np.sqrt(d2 / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
            cost[v2 <= w2] = float("inf")  # Current > speed -> infeasible path
            total_cost = np.sum(cost, axis=1)
        else:  # We must navigate the path in a fixed time
            T = travel_time / (L - 1)
            cost = ((delta_x / T - uinterp) ** 2 + (delta_y / T - vinterp) ** 2) / 2
            total_cost = np.sum(cost, axis=1) * T
        return total_cost

    ### Minimize
    start = time.time()

    x0 = np.linspace(src, dst, K).flatten()  # initial solution
    sigma0 = np.linalg.norm(
        dst - src
    )  # initial standard deviation to sample new solutions
    es = cma.CMAEvolutionStrategy(
        x0, sigma0, inopts={"popsize": popsize, "tolfun": tolfun} | kwargs
    )
    while not es.stop():
        X = es.ask()  # sample len(X) candidate solutions
        es.tell(
            X,
            cost_function(
                control_to_curve(np.array(X)),
            ),
        )
        if verbose:
            es.disp()
    if verbose:
        print("Optimization time:", time.time() - start)
        print("Fuel cost:", es.best.f)
    return control_to_curve(es.best.x[None, :])[0, ...]
