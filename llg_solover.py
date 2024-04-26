import numpy as np
import matplotlib.pyplot as plt

def llg_equation(t, M, H, alpha, gamma):
  """
  Calculates the derivatives of the magnetization vector (M) according to the LLG equation.

  Args:
      t: Current time step.
      M: Magnetization vector (3D numpy array).
      H: Effective field vector (3D numpy array).
      alpha: Damping constant.
      gamma: Gyromagnetic ratio.

  Returns:
      A numpy array containing the derivatives of M (dM/dt).
  """

  dM_dt = -gamma * np.cross(M, H) - alpha * gamma * np.cross(M, np.cross(M, H))
  return dM_dt

def rk5_step(t, M, H, alpha, gamma, dt):
  """
  Performs a single RK5 integration step to update the magnetization vector.

  Args:
      t: Current time step.
      M: Current magnetization vector (3D numpy array).
      H: Effective field vector (3D numpy array).
      alpha: Damping constant.
      gamma: Gyromagnetic ratio.
      dt: Time step size.

  Returns:
      The updated magnetization vector (M_new) after the RK5 step.
  """

  k1 = -1*llg_equation(t, M, H, alpha, gamma)
  k2 = -1*llg_equation(t + dt/4, M + dt/4 * k1, H, alpha, gamma)
  k3 = -1*llg_equation(t + 3*dt/8, M + 3*dt/8 * k2, H, alpha, gamma)
  k4 = -1*llg_equation(t + 12*dt/13, M + 12*dt/13 * k3, H, alpha, gamma)
  k5 = -1*llg_equation(t + dt, M + dt * k4, H, alpha, gamma)

  M_new = M + (dt * (k1 + 2*k2 + 2*k3 + k4 + k5))/6
  return M_new

def llg_solver(t_start, t_end, M0, H, alpha, gamma, dt):
  """
  Solves the LLG equation using the RK5 method over a specified time range.

  Args:
      t_start: Starting time.
      t_end: Ending time.
      M0: Initial magnetization vector (3D numpy array).
      H: Effective field vector (3D numpy array).
      alpha: Damping constant.
      gamma: Gyromagnetic ratio.
      dt: Time step size.

  Returns:
      A list of time steps (t) and a list of corresponding magnetization vectors (M).
  """

  t = np.arange(t_start, t_end + dt, dt)
  M = [M0]
  for i in range(1, len(t)):
    M.append(rk5_step(t[i-1], M[i-1], H, alpha, gamma, dt))

  return t, M

# Example usage
t_start = 0
t_end = 1*10**(-5)
M0 = np.array([1.0, 0.0, 0.0])  # Initial magnetization
H = np.array([0.0, 1.0, 0.0])  # Effective field
alpha = 0.1  # Damping constant
gamma = 2.0e6  # Gyromagnetic ratio (rad/s/T)
dt = 1e-8  # Time step size

t, M = llg_solver(t_start, t_end, M0, H, alpha, gamma, dt)

Mx = [i[0] for i in M ]
My = [i[1] for i in M ]
Mz = [i[2] for i in M ]

plt.plot(t, Mx, label='Mx')
plt.plot(t, My, label='My')
plt.plot(t, Mz, label='Mz')
plt.xlabel('time')
plt.ylabel('Magnetization')
plt.legend()
plt.title('Magnetization Dynamics - Spherical Nanomagnet')
plt.show()
