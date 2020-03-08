"""
Simple model of standing postural stability, consisting of foot and body segments,
and two muscles that create moments about the ankles, tibialis anterior and soleus.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g
from scipy.integrate import solve_ivp
from musculoskeletal import HillTypeMuscle, get_velocity, force_length_tendon


def soleus_length(theta):
    """
    :param theta: body angle (up from prone horizontal)
    :return: soleus length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, .03])
    insertion = [-.05, -.02]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def tibialis_length(theta):
    """
    :param theta: body angle (up from prone horizontal)
    :return: tibialis anterior length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, -.03])
    insertion = [.06, -.03]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def gravity_moment(theta):
    """
    :param theta: angle of body segment (up from prone)
    :return: moment about ankle due to force of gravity on body
    """
    mass = 75 # body mass (kg; excluding feet)
    centre_of_mass_distance = 1 # distance from ankle to body segment centre of mass (m)
    g = 9.81 # acceleration of gravity
    return mass * g * centre_of_mass_distance * np.sin(theta - np.pi / 2)


def dynamics(x, soleus, tibialis, control):
    """
    :param x: state vector (ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length)
    :param soleus: soleus muscle (HillTypeModel)
    :param tibialis: tibialis anterior muscle (HillTypeModel)
    :param control: True if balance should be controlled
    :return: derivative of state vector
    """
    x1, x2, x3, x4 = map(float, x)

    if control:
        angle_diff = abs(np.pi/2 - x1)
        if x1 > np.pi/2:
            a_s = angle_diff - x2
            a_ta = 10 * angle_diff + 10 * x2
        elif x1 < np.pi/2:
            a_s = 5 * angle_diff - 5 * x2
            a_ta = angle_diff - x2
        else:
            a_s = 0
            a_ta = 0
        
        if a_s > 1:
            a_s = 1
        elif a_s < 0:
            a_s = 0

        if a_ta > 1:
            a_ta = 1
        elif a_ta < 0:
            a_ta = 0

    else:
         a_s = 0.05
         a_ta = 0.4
       
    fom_s = soleus.f0M  # N
    fom_ta = tibialis.f0M  # N
    l_s_norm = soleus.norm_tendon_length(soleus_length(x1), x3)
    l_ta_norm = tibialis.norm_tendon_length(tibialis_length(x1), x4)
    d_s = 0.05
    d_ta = 0.03
    i_ankle = 90  # kg*m^2

    tau_s = fom_s * force_length_tendon(l_s_norm) * d_s
    tau_ta = fom_ta * force_length_tendon(l_ta_norm) * d_ta

    x1_dot = x2

    x2_dot = (tau_s - tau_ta + gravity_moment(x1)) / i_ankle

    x3_dot = get_velocity(a_s, x3, l_s_norm)

    x4_dot = get_velocity(a_ta, x4, l_ta_norm)

    return [x1_dot, x2_dot, x3_dot, x4_dot]


def simulate(control, T):
    """
    Runs a simulation of the model and plots results.
    :param control: True if balance should be controlled
    :param T: total time to simulate, in seconds
    """
    rest_length_soleus = soleus_length(np.pi/2)
    rest_length_tibialis = tibialis_length(np.pi/2)

    soleus = HillTypeMuscle(16000, .6*rest_length_soleus, .4*rest_length_soleus)
    tibialis = HillTypeMuscle(2000, .6*rest_length_tibialis, .4*rest_length_tibialis)

    def f(t, x):
        return dynamics(x, soleus, tibialis, control)

    sol = solve_ivp(f, [0, T], [np.pi/2-.001, 0, 1, 1], rtol=1e-5, atol=1e-8)
    time = sol.t
    theta = sol.y[0,:]
    soleus_norm_length_muscle = sol.y[2,:]
    tibialis_norm_length_muscle = sol.y[3,:]

    soleus_moment_arm = .05
    tibialis_moment_arm = .03
    soleus_moment = []
    tibialis_moment = []
    for th, ls, lt in zip(theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
        soleus_moment.append(soleus_moment_arm * soleus.get_force(soleus_length(th), ls))
        tibialis_moment.append(-tibialis_moment_arm * tibialis.get_force(tibialis_length(th), lt))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, sol.y[0,:])
    plt.ylabel('Body angle (rad)')
    plt.subplot(2,1,2)
    plt.plot(time, soleus_moment, 'r')
    plt.plot(time, tibialis_moment, 'g')
    plt.plot(time, gravity_moment(sol.y[0,:]), 'k')
    plt.legend(('soleus', 'tibialis', 'gravity'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torques (Nm)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ############## Question 4 ##############
    simulate(False, 5)

    ############## Question 5 ##############
    simulate(True, 10)