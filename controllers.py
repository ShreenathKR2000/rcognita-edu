"""
Contains controllers a.k.a. agents.

"""

from utilities import dss_sim
from utilities import rep_mat
from utilities import uptria2vec
from utilities import push_vec
import models
import numpy as np
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from scipy.stats import multivariate_normal
from numpy.linalg import lstsq
from numpy import reshape
import warnings
import math
# For debugging purposes
from tabulate import tabulate
import os
from scipy.linalg import solve_discrete_are

def ctrl_selector(t, observation, action_manual, ctrl_nominal, ctrl_benchmarking, ctrl_mode, x_ref=None):
    if ctrl_mode == 'N_CTRL' or ctrl_mode == 'nominal':
        return ctrl_nominal.compute_action(t, observation, x_ref)
    elif ctrl_mode == 'LQR':
        return ctrl_benchmarking.compute_action(t, observation, x_ref)
    elif ctrl_mode == 'MPC':
        return ctrl_benchmarking.compute_action(t, observation, x_ref)
    else:
        raise ValueError(f"Unknown ctrl_mode '{ctrl_mode}'. Choose 'N_CTRL', 'LQR', or 'MPC'.")

    
# -------------------------------------------------------------------------
# Kinematic nominal controller
# -------------------------------------------------------------------------
class N_CTRL:
    def __init__(self, k_rho, k_alpha, k_beta, ctrl_bnds, t0, sampling_time):
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta
        self.ctrl_bnds = ctrl_bnds if ctrl_bnds is not None else np.array([[0, 1.0], [-1.0, 1.0]])
        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        self.action_curr = np.zeros(2)
        
    def compute_action(self, t, observation, x_ref=None):
        time_in_sample = t - self.ctrl_clock
        if time_in_sample >= self.sampling_time:
            self.ctrl_clock = t
            x, y, theta, x_ref, y_ref, theta_ref = observation

            dx = x_ref - x
            dy = y_ref - y

            rho = np.hypot(dx, dy)
            alpha = np.arctan2(dy, dx) - theta
            beta = theta_ref - theta - alpha

            alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
            beta = (beta + np.pi) % (2 * np.pi) - np.pi

            v = self.k_rho * rho
            omega = self.k_alpha * alpha + self.k_beta * beta

            v = np.clip(v, self.ctrl_bnds[0, 0], self.ctrl_bnds[0, 1])
            omega = np.clip(omega, self.ctrl_bnds[1, 0], self.ctrl_bnds[1, 1])

            self.action_curr = np.array([v, omega])

        return self.action_curr
    
    def receive_sys_state(self, state):
        pass

    def upd_accum_obj(self, observation, action):
        pass

    def reset(self, t0):
        self.ctrl_clock = t0
        self.action_curr = np.zeros(2)
    
    def run_obj(self, observation, action):
        return 0.0

# -------------------------------------------------------------------------
# Discrete-time LQR controller
# -------------------------------------------------------------------------
class ControllerLQR:
    """
    Discrete-time LQR controller for 3-wheeled robot with consistent API.
    """
    def __init__(self, A, B, Q, R, sampling_time):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.sampling_time = sampling_time
        P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

    def compute_action(self, t, observation, x_ref=None):
        if x_ref is None:
            x_ref = np.zeros_like(observation)
        state_error = observation - x_ref
        u = -self.K @ state_error
        return u
    
    def run_obj(self, observation, action, x_ref=None):
        if x_ref is None:
            x_ref = np.zeros_like(observation)
        dx = observation - x_ref
        cost = dx.T @ self.Q @ dx + action.T @ self.R @ action
        return cost

class ControllerOptimalPredictive:
    """
    Pure MPC base: optimizes a finite-horizon quadratic cost
    J = sum_{k=0}^{N-1} x_k^T Q x_k + u_k^T R u_k
    using simple Euler prediction and SLSQP solver.

    Parameters
    ----------
    dim_input : int
        Number of inputs (e.g. 2 for [v,omega])
    dim_output : int
        Number of output states (e.g. 3 for [x,y,theta])
    Q : array_like
        State weighting matrix (dim_output x dim_output)
    R : array_like
        Input weighting matrix (dim_input x dim_input)
    N : int
        Prediction horizon length
    dt : float
        Sampling time
    sys_model : object
        System instance with ._state_dyn(t, state, action, []) and .out(state)
    bounds : array_like
        Control bounds shape (dim_input, 2)
    """
    def __init__(self, dim_input, dim_output, Q, R, N, dt, sys_model, bounds):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.N = N
        self.dt = dt
        self.sys_rhs = sys_model._state_dyn
        self.sys_out = sys_model.out
        # flatten bounds for optimization
        lb = np.tile(bounds[:,0], self.N)
        ub = np.tile(bounds[:,1], self.N)
        self.bounds = sp.optimize.Bounds(lb, ub, keep_feasible=True)
        # initial guess
        self.u0 = np.zeros((self.N*self.dim_input,))

    def run_obj(self, x, u):
        # stage cost
        return x.T @ self.Q @ x + u.T @ self.R @ u

    def _cost(self, u_seq, x0):
        # simulate over horizon
        us = u_seq.reshape(self.N, self.dim_input)
        x = x0.copy()
        J = 0.0
        for k in range(self.N):
            uk = us[k]
            J += self.run_obj(x, uk)
            x = x + self.dt * self.sys_rhs(0, x, uk, [])
        return J

    def compute_action(self, t, x0, x_ref=None):
        # optimize sequence but only use first input
        res = minimize(
            lambda u: self._cost(u, x0 - (x_ref if x_ref is not None else np.zeros_like(x0))),
            self.u0,
            method='SLSQP',
            bounds=self.bounds,
            options={'maxiter':50, 'ftol':1e-3}
        )
        u_opt = res.x[:self.dim_input]
        # shift guess for warm start
        self.u0 = np.roll(res.x, -self.dim_input)
        self.u0[-self.dim_input:] = 0
        return u_opt

    def receive_sys_state(self, state):
        pass

    def upd_accum_obj(self, observation, action):
        pass

    def reset(self, t0):
        pass


# -------------------------------------------------------------------------
# MPC controller wrapper
# -------------------------------------------------------------------------
from casadi import SX, vertcat, Function, nlpsol
import numpy as np

class ControllerMPC:
    """
    Nonlinear MPC controller for 3-wheeled robot with 5 states:
    [x, y, theta, v, omega] using CasADi + IPOPT.

    Assumes sys_model has casadi_dynamics(x, u, Ts) function for 1-step prediction.
    """
    def __init__(self, N=20, Q=None, R=None, Qf=None, t0=0.0, sampling_time=0.1, sys_model=None):
        self.N = N
        self.Q = Q if Q is not None else np.diag([5, 5, 0.1, 1, 1])
        self.R = R if R is not None else np.diag([0.1, 0.1])
        self.Qf = Qf if Qf is not None else np.diag([10, 10, 0.5, 2, 2])
        self.Ts = sampling_time
        self.t0 = t0
        self.ctrl_bnds = np.array([[0.0, 1.0], [-1.0, 1.0]])  # [v, omega] bounds
        self.sys_model = sys_model

        self.nx = 5
        self.nu = 2
        self._build_optimizer()
        self.nlp_g = self.g.shape[0]

    def _build_optimizer(self):
        nx = self.nx
        nu = self.nu

        x = SX.sym("x", nx)
        u = SX.sym("u", nu)

        # Use system-provided CasADi dynamics
        if self.sys_model is None or not hasattr(self.sys_model, "casadi_dynamics"):
            raise ValueError("sys_model with casadi_dynamics(x, u, Ts) required for MPC")
        x_next = self.sys_model.casadi_dynamics(x, u, self.Ts)
        self.f = Function("f", [x, u], [x_next])

        # Decision variables
        X = SX.sym("X", nx, self.N + 1)
        U = SX.sym("U", nu, self.N)
        P = SX.sym("P", nx * 2)  # [initial state, reference state]

        x0 = P[:nx]
        x_ref = P[nx:]

        cost = 0
        g = [X[:, 0] - x0]  # Initial condition

        for k in range(self.N):
            xk = X[:, k]
            uk = U[:, k]
            cost += (xk - x_ref).T @ self.Q @ (xk - x_ref) + uk.T @ self.R @ uk
            g.append(X[:, k + 1] - self.f(xk, uk))

        cost += (X[:, self.N] - x_ref).T @ self.Qf @ (X[:, self.N] - x_ref)
        g.append(X[:, self.N] - x_ref)
        OPT_variables = vertcat(U.reshape((-1, 1)), X.reshape((-1, 1)))
        g = vertcat(*g)

        nlp = {"f": cost, "x": OPT_variables, "g": g, "p": P}
        opts = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.max_iter": 300,
            "ipopt.tol": 1e-5
        }

        self.solver = nlpsol("solver", "ipopt", nlp, opts)
        self.g = g

    def compute_action(self, t, observation, x_ref=None):
        x = observation[:self.nx]
        if x_ref is None:
            x_ref = np.zeros(self.nx)
        # STOP logic inside controller as well
        if np.linalg.norm(x[:2] - x_ref[:2]) < 0.05:
            return np.zeros(self.nu)

        p = np.concatenate([x, x_ref])

        u0 = 0.5 * np.ones((self.nu * self.N, 1))
        x0 = np.tile(x.reshape(-1, 1), (1, self.N + 1)).reshape((-1, 1))

        eps = 1e-2  # small tolerance for terminal constraint
        lbg = np.zeros(self.nlp_g)
        ubg = np.zeros(self.nlp_g)
        lbg[-self.nx:] = -eps  # relax terminal constraint slightly
        ubg[-self.nx:] = eps
        try:
            sol = self.solver(x0=np.vstack([u0, x0]), p=p, lbg=lbg, ubg=ubg)
            u_opt = sol['x'][:self.nu].full().flatten()
        except Exception as e:
            print(f"⚠  MPC solver failed at t={t:.2f}s: {e}")
            return np.array([0.0, 0.0])

        v = np.clip(u_opt[0], *self.ctrl_bnds[0])
        omega = np.clip(u_opt[1], *self.ctrl_bnds[1])
        return np.array([v, omega])

    def run_obj(self, observation, action, x_ref=None):
        if x_ref is None:
            x_ref = np.zeros(self.nx)
        dx = observation[:self.nx] - x_ref[:self.nx]
        return dx.T @ self.Q @ dx + action.T @ self.R @ action

    def receive_sys_state(self, state):
        pass

    def upd_accum_obj(self, observation, action):
        pass

    def reset(self, t0):
        pass