import numpy as np
import math
import matplotlib.pyplot as plt

from library.angle import rot_mat_2d

# Convariance for EKF Simulation

Q = np.diag([
    0.1,
    0.1,
    np.deg2rad(1.0),
    1.0
]) ** 2
R = np.diag([1.0, 1.0])**2

# Simulation Parameters
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)])**2
GPS_NOISE = np.diag([0.5, 0.5])

DT = 0.1 # Tick time
SIM_TIME = 100.0 # Sim time

show_animation = True


def calc_input():
    v = 1.0 # [m/s]
    yawrate = 0.1 # [rad/s]
    u = np.array([[v], [yawrate]], dtype=np.float64)

    return u

def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to the gps
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to the input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud

def motion_model(x, u):

    F = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 0]
    ])

    B = np.array([
        [DT * np.cos(x[2, 0]), 0],
        [DT * np.sin(x[2, 0]), 0],
        [0.0, DT],
        [1.0, 0.0]
    ])

    x = F @ x + B @ u

    return x

def observation_model(x):

    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])

    z =  H @ x

    return z


def jacob_f(x, u):
    # Jacobian motion model
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF

def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    # Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    # Update
    j_H = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = j_H@PPred@j_H.T + R
    K = PPred@j_H.T@np.linalg.inv(S)
    x_next = xPred + K@y
    P_next = (np.eye(len(xEst))-K@j_H)@PPred

    return x_next, P_next

def plot_covariance_ellipse(xEst, PEst):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2*np.pi+0.1, 0.1)
    a = np.sqrt(eigval[bigind])
    b = np.sqrt(eigval[smallind])
    x = [a * np.cos(it) for it in t]
    y = [b * np.sin(it) for it in t]
    angle = np.arctan2(eigvec[1, bigind], eigvec[0, bigind])
    fx = rot_mat_2d(angle) @ np.array([x, y])
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main():
    print("Starto !!!")

    time = 0.0

    # State Vector [x, y, yaw v]
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR =  np.zeros((4, 1))

    # History
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xDR
    hz = np.zeros((2, 2))

    while SIM_TIME >= time:
        time += DT

        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history

        hxEst = np.hstack([hxEst, xEst])
        hxDR = np.hstack([hxDR, xDR])
        hxTrue = np.hstack([hxTrue, xTrue])
        hz = np.hstack([hz, z])

        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

if __name__ == "__main__":
    main()
