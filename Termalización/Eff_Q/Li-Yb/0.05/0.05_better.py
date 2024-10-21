import numpy as np
import math
from scipy.integrate import solve_ivp
import random
import time
import logging
from multiprocessing import Pool

# Constants
m_ion = 2.8733965E-25  # YB+ ion mass
m_atom = 1.1525E-26    # Li atom mass
C4 = 5.607E-57
C6 = 5E-19 * C4
frf = 2.5 * 10**6
Omega = 2.0 * np.pi * frf
a = -2.982E-4
q = 0.05
T_ion = 0
T_atom = 0.1E-6
r0 = 0.1E-6
ti = 0.0
tf = 200E-6
dt = 10000
ptol = 1e-10
kB = 1.38064E-23
e = -1.6E-19
Eff = (0.01 * e) / m_ion

logging.basicConfig(level=logging.INFO)

# Differential equations for the system
def system(t, y, a, q, Omega, C4, C6, m_ion, m_atom):
    ionx, iony, ionz, ionvx, ionvy, ionvz, atomx, atomy, atomz, atomvx, atomvy, atomvz = y
    dydt = np.zeros_like(y)

    # Velocities of the ion
    dydt[0], dydt[1], dydt[2] = ionvx, ionvy, ionvz
    dydt[6], dydt[7], dydt[8] = atomvx, atomvy, atomvz

    # Potential for the ion
    cos_term = np.cos(Omega * t)
    dydt[3] = -(a + 2 * q * cos_term) * ionx * (Omega**2 / 4) + Eff
    dydt[4] = -(a + 2 * q * cos_term) * iony * (Omega**2 / 4) + Eff
    dydt[5] = -(2 * a + 2 * -q * cos_term) * ionz * (Omega**2 / 4) + Eff

    # Atom-ion interaction potential
    r = np.sqrt((atomx - ionx)**2 + (atomy - iony)**2 + (atomz - ionz)**2)
    force = -(2 * C4 / r**5 - 6 * C6 / r**7)
    r_inv = 1 / r
    dydt[3] += force * (ionx - atomx) * r_inv / m_ion
    dydt[4] += force * (iony - atomy) * r_inv / m_ion
    dydt[5] += force * (ionz - atomz) * r_inv / m_ion
    dydt[9] += force * (atomx - ionx) * r_inv / m_atom
    dydt[10] += force * (atomy - iony) * r_inv / m_atom
    dydt[11] += force * (atomz - ionz) * r_inv / m_atom

    return dydt

# Position on sphere
def position_on_sphere(radius):
    phi = 2 * np.pi * np.random.rand()
    theta = np.arccos(2 * np.random.rand() - 1)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z

# Velocities for atom and ion
def velocidad_atom(temperature, mass):
    v_r = -abs(np.random.weibull(2) * np.sqrt(2 * kB * temperature / mass))
    v_theta = np.sqrt((2 * kB * temperature) / mass) * random.gauss(0, 1)
    v_phi = np.sqrt((2 * kB * temperature) / mass) * random.gauss(0, 1)
    return v_r, v_theta, v_phi

def velocidad_ion(temperature, mass):
    return np.sqrt((2 * kB * temperature) / mass) * random.gauss(0, 1)

# Event to stop integration
def event_condition(t, y):
    atomx, atomy, atomz = y[6], y[7], y[8]
    return np.sqrt(atomx**2 + atomy**2 + atomz**2) - 2.0E-6

event_condition.terminal = True
event_condition.direction = 1

# Main simulation for each process
def run_simulation(i):
    ionx_0 = iony_0 = ionz_0 = 0.0
    ionvx_0 = velocidad_ion(T_ion, m_ion)
    ionvy_0 = velocidad_ion(T_ion, m_ion)
    ionvz_0 = velocidad_ion(T_ion, m_ion)
    atomx_0, atomy_0, atomz_0 = position_on_sphere(r0)
    atomvx_0, atomvy_0, atomvz_0 = velocidad_atom(T_atom, m_atom)

    y0 = [ionx_0, iony_0, ionz_0, ionvx_0, ionvy_0, ionvz_0, atomx_0, atomy_0, atomz_0, atomvx_0, atomvy_0, atomvz_0]

    results = []
    for _ in range(num_collisions):
        sol = solve_ivp(system, (ti, tf), y0, t_eval=np.linspace(ti, tf, dt), method='RK45',
                        args=(a, q, Omega, C4, C6, m_ion, m_atom), rtol=ptol, events=event_condition)

        # Extract energy only to save memory
        ionvx_sol = sol.y[3]
        ionvy_sol = sol.y[4]
        ionvz_sol = sol.y[5]
        E_kin = 0.5 * m_ion * (ionvx_sol**2 + ionvy_sol**2 + ionvz_sol**2)

        results.append(np.mean(E_kin))

        # Update atom position and velocity for next collision
        atomx_0, atomy_0, atomz_0 = position_on_sphere(r0)
        atomvx_0, atomvy_0, atomvz_0 = velocidad_atom(T_atom, m_atom)
        y0 = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], ionvx_sol[-1], ionvy_sol[-1], ionvz_sol[-1], atomx_0, atomy_0, atomz_0, atomvx_0, atomvy_0, atomvz_0]

    return results

if __name__ == '__main__':
    logging.info("Inicio de la ejecución")
    start_time = time.time()

    num_cpus = 24
    num_solution = 10  # Define cuántas simulaciones paralelas se ejecutarán
    num_collisions = 500

    with Pool(processes=num_cpus) as pool:
        all_results = pool.map(run_simulation, range(num_solution))

    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    logging.info(f"Tiempo total de ejecución con {num_cpus} núcleos: {execution_time:.2f} minutos")

    # Post-process results (e.g., average kinetic energy across collisions)
    Ekin_total = [np.mean(sim_result) for sim_result in all_results]
    print(f"Energía cinética media total: {np.mean(Ekin_total)}")
