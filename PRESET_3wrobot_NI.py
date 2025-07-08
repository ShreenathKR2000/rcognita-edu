#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# adjust these imports to your package structure:
from systems     import Sys3WRobotNI, Sys3WRobot
from controllers import N_CTRL, ControllerLQR, ControllerMPC

def run_kinematic(dt, Tfinal, x0, x_goal):
    gain_sets = [
        {"k_rho": 0.5, "k_alpha": 1.5, "k_beta": -0.5},
        {"k_rho": 0.7, "k_alpha": 2.0, "k_beta": -0.8},
        {"k_rho": 0.9, "k_alpha": 2.5, "k_beta": -1.0},
        {"k_rho": 1.1, "k_alpha": 3.0, "k_beta": -1.2},
        {"k_rho": 1.3, "k_alpha": 3.5, "k_beta": -1.5},
        {"k_rho": 1.5, "k_alpha": 4.0, "k_beta": -1.7},
        {"k_rho": 1.7, "k_alpha": 4.5, "k_beta": -2.0},
        {"k_rho": 1.9, "k_alpha": 5.0, "k_beta": -2.2},
        {"k_rho": 2.1, "k_alpha": 5.5, "k_beta": -2.5},
        {"k_rho": 2.3, "k_alpha": 6.0, "k_beta": -2.8},
    ]

    t_vec = np.arange(0, Tfinal+dt, dt)
    all_X = []
    all_err = []
    all_U = []

    for i, gains in enumerate(gain_sets, start=1):
        robot = Sys3WRobotNI(sys_type='discr_fnc',
                             dim_state=3, dim_input=2,
                             dim_output=3, dim_disturb=0,
                             ctrl_bnds=np.array([[0,1],[-1,1]]))
        ctrl  = N_CTRL(k_rho=gains["k_rho"],
                       k_alpha=gains["k_alpha"],
                       k_beta=gains["k_beta"],
                       ctrl_bnds=robot.ctrl_bnds,
                       t0=0.0, sampling_time=dt)

        X   = np.zeros((len(t_vec),3))
        U   = np.zeros((len(t_vec)-1,2))
        err = np.zeros(len(t_vec))

        x = x0.copy()
        for k, t in enumerate(t_vec):
            X[k] = x
            obs = np.concatenate([x, x_goal])
            u   = ctrl.compute_action(t, obs)
            if k < len(t_vec)-1:
                U[k]   = u
                x      = robot.integrate(x, u, t, dt)
                err[k] = np.linalg.norm(x[:2] - x_goal[:2])

        all_X.append(X)
        all_err.append(err)
        all_U.append(U)

    # Combined plots
    os.makedirs("kinematic_results", exist_ok=True)

    # Trajectories
    plt.figure()
    for i, X in enumerate(all_X, start=1):
        plt.plot(X[:,0], X[:,1], label=f"Sim {i}")
    plt.plot(x_goal[0], x_goal[1], 'rx', label="Goal")
    plt.title("Kinematic Controller: Trajectories")
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.grid(); plt.legend()
    plt.savefig("kinematic_results/all_trajectories.png")
    plt.close()

    # Errors
    plt.figure()
    for i, err in enumerate(all_err, start=1):
        plt.plot(t_vec, err, label=f"Sim {i}")
    plt.title("Kinematic Controller: Tracking Error")
    plt.xlabel("time [s]"); plt.ylabel("||e|| [m]"); plt.grid(); plt.legend()
    plt.savefig("kinematic_results/all_errors.png")
    plt.close()

    # Control inputs
    plt.figure()
    for i, U in enumerate(all_U, start=1):
        plt.plot(t_vec[:-1], U[:,0],   label=f"v Sim {i}")
        plt.plot(t_vec[:-1], U[:,1],   '--', label=f"ω Sim {i}")
    plt.title("Kinematic Controller: Control Inputs")
    plt.xlabel("time [s]"); plt.ylabel("input"); plt.grid()
    # to avoid huge legend, only label first few
    plt.legend(ncol=2, fontsize='small')
    plt.savefig("kinematic_results/all_controls.png")
    plt.close()


def run_lqr(dt, Tfinal, x0, x_goal):
    lqr_sets = [
        (np.diag([1, 1, 0.1]), np.diag([0.5, 0.5])),
        (np.diag([2, 2, 0.2]), np.diag([0.4, 0.4])),
        (np.diag([3, 3, 0.3]), np.diag([0.3, 0.3])),
        (np.diag([4, 4, 0.4]), np.diag([0.25, 0.25])),
        (np.diag([5, 5, 0.5]), np.diag([0.2, 0.2])),
        (np.diag([6, 6, 0.6]), np.diag([0.15, 0.15])),
        (np.diag([7, 7, 0.7]), np.diag([0.1, 0.1])),
        (np.diag([8, 8, 0.8]), np.diag([0.08, 0.08])),
        (np.diag([9, 9, 0.9]), np.diag([0.05, 0.05])),
        (np.diag([10, 10, 1.0]), np.diag([0.01, 0.01])),
    ]

    t_vec = np.arange(0, Tfinal+dt, dt)
    all_X   = []
    all_err = []
    all_cost= []

    # linearized matrices at θ=0
    A = np.array([[1,0,-dt],[0,1,dt],[0,0,1]])
    B = np.array([[dt,0],[0,0],[0,dt]])

    for i, (Q, R) in enumerate(lqr_sets, start=1):
        ctrl = ControllerLQR(A, B, Q, R, sampling_time=dt)

        X   = np.zeros((len(t_vec),3))
        err = np.zeros(len(t_vec))
        cost_accum = np.zeros(len(t_vec))

        x = x0.copy()
        J = 0.0
        for k, t in enumerate(t_vec):
            X[k] = x
            u    = ctrl.compute_action(t, x, x_goal)
            u[0] = np.clip(u[0], 0, 1)
            u[1] = np.clip(u[1], -1, 1)
            if k < len(t_vec)-1:
                # unicycle kinematics
                x[0] += dt*u[0]*np.cos(x[2])
                x[1] += dt*u[0]*np.sin(x[2])
                x[2] += dt*u[1]
                e       = x - x_goal
                err[k]  = np.linalg.norm(e[:2])
                J      += e.T@Q@e + u.T@R@u
                cost_accum[k] = J

        all_X.append(X)
        all_err.append(err)
        all_cost.append(cost_accum)

    os.makedirs("lqr_results", exist_ok=True)

    # Trajectories
    plt.figure()
    for i, X in enumerate(all_X, start=1):
        plt.plot(X[:,0], X[:,1], label=f"Sim {i}")
    plt.plot(x_goal[0], x_goal[1], 'rx', label="Goal")
    plt.title("LQR: Trajectories")
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.grid(); plt.legend()
    plt.savefig("lqr_results/all_trajectories.png"); plt.close()

    # Errors
    plt.figure()
    for i, err in enumerate(all_err, start=1):
        plt.plot(t_vec, err, label=f"Sim {i}")
    plt.title("LQR: Tracking Error")
    plt.xlabel("time [s]"); plt.ylabel("||e|| [m]"); plt.grid(); plt.legend()
    plt.savefig("lqr_results/all_errors.png"); plt.close()

    # Accumulated Cost
    plt.figure()
    for i, cost in enumerate(all_cost, start=1):
        plt.plot(t_vec, cost, label=f"Sim {i}")
    plt.title("LQR: Accumulated Cost")
    plt.xlabel("time [s]"); plt.ylabel("cost"); plt.grid(); plt.legend()
    plt.savefig("lqr_results/all_costs.png"); plt.close()


def run_mpc(dt, Tfinal, x0, x_goal):
    mpc_sets = [
        {"N": 20, "Q": np.diag([10, 10, 2, 20, 20]),  "R": np.diag([0.4, 0.4]),  "Qf": np.diag([50, 50, 10, 100, 100])},
        {"N": 25, "Q": np.diag([12, 12, 2, 25, 25]),  "R": np.diag([0.35, 0.35]),"Qf": np.diag([60, 60, 12, 120, 120])},
        {"N": 32, "Q": np.diag([15, 15, 3, 33, 33]),  "R": np.diag([0.3, 0.3]),  "Qf": np.diag([75, 75, 15, 150, 150])},
        {"N": 35, "Q": np.diag([16, 16, 3, 35, 35]),  "R": np.diag([0.25, 0.25]),"Qf": np.diag([80, 80, 15, 160, 160])},
        {"N": 40, "Q": np.diag([18, 18, 4, 40, 40]),  "R": np.diag([0.2, 0.2]),  "Qf": np.diag([90, 90, 20, 180, 180])},
        {"N": 45, "Q": np.diag([20, 20, 4, 45, 45]),  "R": np.diag([0.18, 0.18]),"Qf": np.diag([100, 100, 20, 200, 200])},
        {"N": 50, "Q": np.diag([22, 22, 5, 50, 50]),  "R": np.diag([0.15, 0.15]),"Qf": np.diag([110, 110, 25, 220, 220])},
        {"N": 55, "Q": np.diag([24, 24, 5, 55, 55]),  "R": np.diag([0.12, 0.12]),"Qf": np.diag([120, 120, 25, 240, 240])},
        {"N": 60, "Q": np.diag([26, 26, 6, 60, 60]),  "R": np.diag([0.1, 0.1]),  "Qf": np.diag([130, 130, 30, 260, 260])},
        {"N": 60, "Q": np.diag([28, 28, 6, 60, 60]),  "R": np.diag([0.08, 0.08]),"Qf": np.diag([140, 140, 30, 280, 280])},
    ]

    t_vec = np.arange(0, Tfinal+dt, dt)
    all_X   = []
    all_err = []
    all_cost= []
    all_t = []  # to store truncated time vectors per sim
    u = np.zeros(2) 
    
    for i, params in enumerate(mpc_sets, start=1):
        # Use full 5-state robot model, discrete-time
        robot = Sys3WRobot(sys_type='discr_fnc',
                           dim_state=5, dim_input=2,
                           dim_output=5, dim_disturb=0,
                           pars=[1.0, 0.1],
                           ctrl_bnds=np.array([[0,1],[-1,1]]))
        ctrl  = ControllerMPC(N=params["N"],
                              Q=params["Q"], R=params["R"],
                              Qf=params["Qf"],
                              sampling_time=dt,
                              sys_model=robot)

        X   = np.zeros((len(t_vec),3))  # only store first 3 states (x,y,theta) for plotting
        x_goal_5 = np.array([x_goal[0], x_goal[1], x_goal[2], 0.0, 0.0])  # Target 5D state
        err = np.zeros(len(t_vec))
        cost_accum = np.zeros(len(t_vec))

        # Initialize 5D state vector: [x, y, theta, v, omega]
        x5 = np.array([x0[0], x0[1], x0[2], 0.0, 0.0])
        J = 0.0
        k_stop = len(t_vec) - 1  # default to full length if goal not reached
        
        for k, t in enumerate(t_vec):
            X[k] = x5[:3]
            dist = np.linalg.norm(x5[:2] - x_goal[:2])

            err[k] = dist                 # assign error here
            cost_accum[k] = J
            # STOP if within tolerance
            if dist < 0.3:
                print(f"✅ Robot reached goal in Sim {i} at t={t:.2f}s")
                k_stop = k
                break

            # normal MPC update
            u = ctrl.compute_action(t, x5, x_goal_5)
            x5 = robot.integrate(x5, u, t, dt)
            J += ctrl.run_obj(x5, u, x_goal_5)

        # Truncate arrays and time vector up to stopping point
        X = X[:k_stop+1]
        err = err[:k_stop+1]
        cost_accum = cost_accum[:k_stop+1]
        t_plot = t_vec[:k_stop+1]

        all_X.append(X)
        all_err.append(err)
        all_cost.append(cost_accum)
        all_t.append(t_plot)   # save corresponding time vector

    os.makedirs("mpc_results", exist_ok=True)

    # Trajectories
    plt.figure()
    for i, X in enumerate(all_X, start=1):
        plt.plot(X[:,0], X[:,1], label=f"Sim {i}")
    plt.plot(x_goal[0], x_goal[1], 'rx', label="Goal")
    plt.title("MPC (5-state): Trajectories")
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.grid(); plt.legend()
    plt.savefig("mpc_results/all_trajectories.png"); plt.close()

        # Errors
    plt.figure()
    for i, (err, t_plot) in enumerate(zip(all_err, all_t), start=1):
        plt.plot(t_plot, err, label=f"Sim {i}")  # Use correct time vector per err
    plt.title("MPC (5-state): Tracking Error")
    plt.xlabel("time [s]"); plt.ylabel("||e|| [m]"); plt.grid(); plt.legend()
    plt.savefig("mpc_results/all_errors.png")
    plt.close()

    # Accumulated Cost
    plt.figure()
    for i, (cost, t_plot) in enumerate(zip(all_cost, all_t), start=1):
        plt.plot(t_plot, cost, label=f"Sim {i}")
    plt.title("MPC (5-state): Accumulated Cost")
    plt.xlabel("time [s]"); plt.ylabel("cost"); plt.grid(); plt.legend()
    plt.savefig("mpc_results/all_costs.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark 3‑wheel robot controllers"
    )
    parser.add_argument(
        "--ctrl_mode", choices=["N_CTRL","LQR","MPC"],
        default="N_CTRL",
        help="which controller to benchmark"
    )
    args = parser.parse_args()

    # common initial/goal
    dt     = 0.1
    Tfinal = 20.0
    x0     = np.array([0.0,0.0,0.0])
    x_goal = np.array([2.0,2.0,0.0])

    if args.ctrl_mode == "N_CTRL":
        run_kinematic(dt, Tfinal, x0, x_goal)
    elif args.ctrl_mode == "LQR":
        run_lqr(dt, Tfinal, x0, x_goal)
    elif args.ctrl_mode == "MPC":
        run_mpc(dt, Tfinal, x0, x_goal)

    print(f"All {args.ctrl_mode} simulations done. Check the corresponding folder.")