#!/usr/bin/env python3
# -- coding: utf-8 --
"""
This module contains a generic interface for systems (environments) as well as concrete systems as realizations of the former

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
from numpy.random import randn
from casadi import vertcat, cos, sin, SX

class System:
    """
    Interface class of dynamical systems a.k.a. environments.
    Concrete systems should be built upon this class.
    To design a concrete system: inherit this class, override:
        | :func:~systems.system._state_dyn :
        | right-hand side of system description (required)
        | :func:~systems.system._disturb_dyn :
        | right-hand side of disturbance model (if necessary)
        | :func:~systems.system._ctrl_dyn :
        | right-hand side of controller dynamical model (if necessary)
        | :func:~systems.system.out :
        | system out (if not overridden, output is identical to state)
      
    Attributes
    ----------
    sys_type : : string
        Type of system by description:
            
        | `diff_eqn` : differential equation :math:\mathcal D state = f(state, action, disturb)
        | `discr_fnc` : difference equation :math:state^+ = f(state, action, disturb)
        | `discr_prob` :  by probability distribution :math:X^+ \sim P_X(state^+| state, action, disturb)
    
    where:
        
        | :math:state : state
        | :math:action : input
        | :math:disturb : disturbance
        
    The time variable `t` is commonly used by ODE solvers, and you shouldn't have it explicitly referenced in the definition, unless your system is non-autonomous.
    For the latter case, however, you already have the input and disturbance at your disposal.
    
    Parameters of the system are contained in `pars` attribute.
    
    dim_state, dim_input, dim_output, dim_disturb : : integer
        System dimensions 
    pars : : list
        List of fixed parameters of the system
    ctrl_bnds : : array of shape `[dim_input, 2]`
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default)
    is_dyn_ctrl : : 0 or 1
        If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
    is_disturb : : 0 or 1
        If 0, no disturbance is fed into the system
    pars_disturb : : list
        Parameters of the disturbance model
        
   Each concrete system must realize `System` and define `name` attribute.   
        
    """
    def __init__(self,
                 sys_type,
                 dim_state,
                 dim_input,
                 dim_output,
                 dim_disturb,
                 pars=[],
                 ctrl_bnds=[],
                 is_dyn_ctrl=0,
                 is_disturb=0,
                 pars_disturb=[]):
        
        """
        Parameters
        ----------
        sys_type : : string
            Type of system by description:
                
            | `diff_eqn` : differential equation :math:\mathcal D state = f(state, action, disturb)
            | `discr_fnc` : difference equation :math:state^+ = f(state, action, disturb)
            | `discr_prob` :  by probability distribution :math:X^+ \sim P_X(state^+| state, action, disturb)
        
        where:
            
            | :math:state : state
            | :math:action : input
            | :math:disturb : disturbance
            
        The time variable `t` is commonly used by ODE solvers, and you shouldn't have it explicitly referenced in the definition, unless your system is non-autonomous.
        For the latter case, however, you already have the input and disturbance at your disposal.
        
        Parameters of the system are contained in `pars` attribute.
        
        dim_state, dim_input, dim_output, dim_disturb : : integer
            System dimensions 
        pars : : list
            List of fixed parameters of the system
        ctrl_bnds : : array of shape `[dim_input, 2]`
            Box control constraints.
            First element in each row is the lower bound, the second - the upper bound.
            If empty, control is unconstrained (default)
        is_dyn_ctrl : : 0 or 1
            If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
        is_disturb : : 0 or 1
            If 0, no disturbance is fed into the system
        pars_disturb : : list
            Parameters of the disturbance model        
        """
        
        self.sys_type = sys_type
        
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_disturb = dim_disturb   
        self.pars = pars
        self.ctrl_bnds = ctrl_bnds
        self.is_dyn_ctrl = is_dyn_ctrl
        self.is_disturb = is_disturb
        self.pars_disturb = pars_disturb
        
        # Track system's state
        self._state = np.zeros(dim_state)
        
        # Current input (a.k.a. action)
        self.action = np.zeros(dim_input)
        
        if is_dyn_ctrl:
            if is_disturb:
                self._dim_full_state = self.dim_state + self.dim_disturb + self.dim_input
            else:
                self._dim_full_state = self.dim_state
        else:
            if is_disturb:
                self._dim_full_state = self.dim_state + self.dim_disturb
            else:
                self._dim_full_state = self.dim_state
            
    def _state_dyn(self, t, state, action, disturb):
        """
        Description of the system internal dynamics.
        Depending on the system type, may be either the right-hand side of the respective differential or difference equation, or a probability distribution.
        As a probability disitribution, `_state_dyn` should return a number in :math:[0,1]
        
        """
        pass

    def _disturb_dyn(self, t, disturb):
        """
        Dynamical disturbance model depending on the system type:
            
        | `sys_type = "diff_eqn"` : :math:\mathcal D disturb = f_q(disturb)    
        | `sys_type = "discr_fnc"` : :math:disturb^+ = f_q(disturb)
        | `sys_type = "discr_prob"` : :math:disturb^+ \sim P_Q(disturb^+|disturb)
        
        """       
        pass

    def _ctrl_dyn(self, t, action, observation):
        """
        Dynamical controller. When `is_dyn_ctrl=0`, the controller is considered static, which is to say that the control actions are
        computed immediately from the system's output.
        In case of a dynamical controller, the system's state vector effectively gets extended.
        Dynamical controllers have some advantages compared to the static ones.
        
        Depending on the system type, can be:
            
        | `sys_type = "diff_eqn"` : :math:\mathcal D action = f_u(action, observation)    
        | `sys_type = "discr_fnc"` : :math:action^+ = f_u(action, observation)  
        | `sys_type = "discr_prob"` : :math:action^+ \sim P_U(action^+|action, observation)        
        
        """
        Daction = np.zeros(self.dim_input)
    
        return Daction 

    def out(self, state, action=[]):
        """
        System output.
        This is commonly associated with signals that are measured in the system.
        Normally, output depends only on state `state` since no physical processes transmit input to output instantly.       
        
        See also
        --------
        :func:~systems.system._state_dyn
        
        """
        # Trivial case: output identical to state
        observation = state
        return observation
    
    def receive_action(self, action):
        """
        Receive exogeneous control action to be fed into the system.
        This action is commonly computed by your controller (agent) using the system output :func:~systems.system.out. 

        Parameters
        ----------
        action : : array of shape `[dim_input, ]`
            Action
            
        """
        self.action = action
        
    def closed_loop_rhs(self, t, state_full):
        """
        Right-hand side of the closed-loop system description.
        Combines everything into a single vector that corresponds to the right-hand side of the closed-loop system description for further use by simulators.
        
        Attributes
        ----------
        state_full : : vector
            Current closed-loop system state        
        
        """
        rhs_full_state = np.zeros(self._dim_full_state)
        
        state = state_full[0:self.dim_state]
        
        if self.is_disturb:
            disturb = state_full[self.dim_state:]
        else:
            disturb = []
        
        if self.is_dyn_ctrl:
            action = state_full[-self.dim_input:]
            observation = self.out(state)
            rhs_full_state[-self.dim_input:] = self._ctrlDyn(t, action, observation)
        else:
            # Fetch the control action stored in the system
            action = self.action
        
        if self.ctrl_bnds.any():
            for k in range(self.dim_input):
                action[k] = np.clip(action[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])
        
        rhs_full_state[0:self.dim_state] = self._state_dyn(t, state, action, disturb)
        
        if self.is_disturb:
            rhs_full_state[self.dim_state:] = self._disturb_dyn(t, disturb)
        
        # Track system's state
        self._state = state
        
        return rhs_full_state    
    
class Sys3WRobotNI(System):
    """
    3-Wheel Robot with dynamics (non-holonomic integrator, unicyclic for kinematic).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '3wrobotNI'
        if self.is_disturb:
            self.sigma_disturb = self.pars_disturb[0]
            self.mu_disturb = self.pars_disturb[1]
            self.tau_disturb = self.pars_disturb[2]

    def _state_dyn(self, t, state, action, disturb=[]):   
        x, y, theta = state
        v, omega = action

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega

        return np.array([dx, dy, dtheta])

    def out(self, state, action=[]):
        return state

    def integrate(self, x, u, t, dt=0.1):
        dx = self._state_dyn(t, x, u)
        return x + dt * dx
    
class Sys3WRobot(System):
    """
    3-Wheel Robot with dynamics (extended non-holonomic double integrator, non unicyclic).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '3wrobot'

    def _state_dyn(self, t, state, action, disturb=[]):
        m, I = self.pars[0], self.pars[1]
        Dstate = np.zeros(self.dim_state)
        Dstate[0] = state[3] * np.cos(state[2])
        Dstate[1] = state[3] * np.sin(state[2])
        Dstate[2] = state[4]
        Dstate[3] = 1 / m * action[0]
        Dstate[4] = 1 / I * action[1]
        return Dstate

    def out(self, state, action=[]):
        return state

    def integrate(self, x, u, t, dt=0.1):
        """
        Simple forward Euler integrator for simulation.
        """
        dx = self._state_dyn(t, x, u)
        x_next = x + dt * dx

        # Auto-stop strategy (friction-like effect)
        velocity_threshold = 1e-3
        if np.abs(x_next[3]) < velocity_threshold:
            x_next[3] = 0.0  # linear velocity
        if np.abs(x_next[4]) < velocity_threshold:
            x_next[4] = 0.0  # angular velocity

        return x_next
    
    def get_A_matrix(self):
        A = np.zeros((5,5))
        theta = self._state[2]
        v = self._state[3]
        A[0,2] = -v * np.sin(theta)
        A[0,3] = np.cos(theta)
        A[1,2] = v * np.cos(theta)
        A[1,3] = np.sin(theta)
        A[2,4] = 1
        # Linear velocity and angular velocity dynamics have no state dependency (integrator part)
        return A

    def get_B_matrix(self):
        m, I = self.pars[0], self.pars[1]
        B = np.zeros((5,2))
        B[3,0] = 1/m
        B[4,1] = 1/I
        return B
    
    def casadi_dynamics(self, x, u, Ts):
        theta = x[2]
        v = x[3]
        omega = x[4]

        # Positions updated by velocities
        x_next = vertcat(
            x[0] + Ts * v * cos(theta),   # x position
            x[1] + Ts * v * sin(theta),   # y position
            x[2] + Ts * omega,            # orientation
            x[3] + Ts * u[0],             # linear velocity updated by acceleration
            x[4] + Ts * u[1]              # angular velocity updated by angular acceleration
        )
        return x_next