import numpy as np
import math
from math import sqrt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count
from scipy.signal import find_peaks

#Ion: Yb+ , atom: Li

#constants
m_ion = 2.8733965E-25  #(YB+) 2,8733965 × 10^-25 kg (SR+ 2.83E-25)
m_atom = 1.1525801E-26 # (LITIO) 1,1525801 × 10^-26 kg (RB 1.44E-25 )
C4=5.607E-57   #C4=5.607E-57   Paper Prospects #C4=1.1E-56 RbSr+
C6=5E-19 * C4   #C6=5.0E-19     Paper Prospects (LIYB+) C6 = 2.803 × 10−75 Jm6 C4 = 2.803 × 10−57
C8= 5.87E-78 #(LITIO) 5.87E-78 (RUBIDIO) 1.94E-77
C12=1.07E-135 #(LITIO) 1.07E-135 (RUBIDIO) 5.40E-134
frf = 2.5 * 10**6
Omega= 2 * np.pi * frf #MHz #Depende tambien mucho de esta fase
a = -2.982E-4  #-3E-7, da un buen comportamiento
q = 0.1     #Entre 0.1 y otro
#a = 0  #-3E-7, da un buen comportamiento
#q = 0     #Entre 0.1 y otro
T_ion =  0 # da un buen comportamiento  0.5 milikelvin   Paper Ziv Meir (de Jesus)
T_atom = 2E-6
#T_atom = 2.0E-6  #0.1E-6, da un buen comportamiento   #6 microkelvin (mucho más frio átomo que ion) Paper Ziv Meir
r0 = 0.15E-6
ti= 0
tf= 50E-6   #o 1.0E-5       #0.001, da un buen comportamiento pal ion #de matlab 200e-6;
dt= 1000
ptol = 1e-10
kB = 1.38064E-23
num_cpus = cpu_count()
num_solution = 5000
num_collisions = 1 #50 colisiones*3= 150 trayectorias que se estudian   500*10= 5000 trayectorias estudiadas
activation_time = 7E-6 #Seran 6, 7, 8 , 9 ,10
umbral = 4E-8
r_umbral = 1.3E-7
t_three_body = 45E-6

def system(t, y, a, q, Omega, C4, C6, m_ion, m_a):
    """Sistema de ecuaciones diferenciales para el ion y dos átomos."""
    (ionx, iony, ionz, ionvx, ionvy, ionvz,
     atom1x, atom1y, atom1z, atom1vx, atom1vy, atom1vz,
     atom2x, atom2y, atom2z, atom2vx, atom2vy, atom2vz) = y

    dydt = np.zeros_like(y)

    # Velocidades del ion
    dydt[0] = ionvx
    dydt[1] = ionvy
    dydt[2] = ionvz

    # Velocidades del átomo 1
    dydt[6] = atom1vx
    dydt[7] = atom1vy
    dydt[8] = atom1vz

    # Velocidades del átomo 2
    if t >= activation_time:
        dydt[12] = atom2vx
        dydt[13] = atom2vy
        dydt[14] = atom2vz

    # Fuerzas de la trampa para el ion
    dydt[3] = -(a + 2 * q * np.cos(Omega * t)) * ionx * (Omega**2 / 4)
    dydt[4] = -(a + 2 * q * np.cos(Omega * t)) * iony * (Omega**2 / 4)
    dydt[5] = -(2 * a + 2 * -q * np.cos(Omega * t)) * ionz * (Omega**2 / 4)

    # Interacción ion-átomo 1
    r1 = np.sqrt((atom1x - ionx)**2 + (atom1y - iony)**2 + (atom1z - ionz)**2)
    dydt[3] += -(2 * C4 / r1**5 - 6 * C6 / r1**7) * (ionx - atom1x) / (m_ion * r1)
    dydt[4] += -(2 * C4 / r1**5 - 6 * C6 / r1**7) * (iony - atom1y) / (m_ion * r1)
    dydt[5] += -(2 * C4 / r1**5 - 6 * C6 / r1**7) * (ionz - atom1z) / (m_ion * r1)

    dydt[9] += -(2 * C4 / r1**5 - 6 * C6 / r1**7) * (atom1x - ionx) / (m_atom * r1)
    dydt[10] += -(2 * C4 / r1**5 - 6 * C6 / r1**7) * (atom1y - iony) / (m_atom * r1)
    dydt[11] += -(2 * C4 / r1**5 - 6 * C6 / r1**7) * (atom1z - ionz) / (m_atom * r1)

    if t >= activation_time:
        # Interacción ion-átomo 2
        r2 = np.sqrt((atom2x - ionx)**2 + (atom2y - iony)**2 + (atom2z - ionz)**2)
        dydt[3] += -(2 * C4 / r2**5 - 6 * C6 / r2**7) * (ionx - atom2x) / (m_ion * r2)
        dydt[4] += -(2 * C4 / r2**5 - 6 * C6 / r2**7) * (iony - atom2y) / (m_ion * r2)
        dydt[5] += -(2 * C4 / r2**5 - 6 * C6 / r2**7) * (ionz - atom2z) / (m_ion * r2)

        dydt[15] += -(2 * C4 / r2**5 - 6 * C6 / r2**7) * (atom2x - ionx) / (m_atom * r2)
        dydt[16] += -(2 * C4 / r2**5 - 6 * C6 / r2**7) * (atom2y - iony) / (m_atom * r2)
        dydt[17] += -(2 * C4 / r2**5 - 6 * C6 / r2**7) * (atom2z - ionz) / (m_atom * r2)

        # Interacción átomo 1-átomo 2
        r12 = np.sqrt((atom2x - atom1x)**2 + (atom2y - atom1y)**2 + (atom2z - atom1z)**2)
        if r12 > 0:  # Evitar división por cero
            dydt[9] += -(6 * C8 / r12**7 - 12 * C12 / r12**13) * (atom1x - atom2x) / (m_atom * r12)
            dydt[10] += -(6 * C8 / r12**7 - 12 * C12 / r12**13) * (atom1y - atom2y) / (m_atom * r12)
            dydt[11] += -(6 * C8 / r12**7 - 12 * C12 / r12**13) * (atom1z - atom2z) / (m_atom * r12)

            dydt[15] -= -(6 * C8 / r12**7 - 12 * C12 / r12**13) * (atom2x - atom1x) / (m_atom * r12)
            dydt[16] -= -(6 * C8 / r12**7 - 12 * C12 / r12**13) * (atom2y - atom1y) / (m_atom * r12)
            dydt[17] -= -(6 * C8 / r12**7 - 12 * C12 / r12**13) * (atom2z - atom1z) / (m_atom * r12)

    return dydt

#Funciones
def position_on_sphere(radius):
    phi = 2 * np.pi * np.random.rand()
    theta = np.arccos(2 * np.random.rand() - 1)
    # Convertir a coordenadas cartesianas
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z

def velocidad_atom(temperature, mass, x, y, z):
    phi = 2 * np.pi * np.random.rand()
    theta = np.arccos(2 * np.random.rand() - 1)

    kb = 1.380649e-23  # Constante de Boltzmann en m^2 kg s^-2 K^-1

    # Spherical velocities
    v_theta = np.sqrt((2 * kb * temperature) / mass) * random.gauss(0, 1)
    v_phi = np.sqrt((2 * kb * temperature) / mass) * random.gauss(0, 1)
    v_r = -abs(np.random.weibull(2) * np.sqrt(2 * kb * temperature / mass))

    # Componentes de velocidad en coordenadas cartesianas
    v_x = v_r * np.sin(theta) * np.cos(phi) + v_theta * np.cos(theta) * np.cos(phi) - v_phi * np.sin(phi)
    v_y = v_r * np.sin(theta) * np.sin(phi) + v_theta * np.cos(theta) * np.sin(phi) + v_phi * np.cos(phi)
    v_z = v_r * np.cos(theta) - v_theta * np.sin(theta)

    return v_x, v_y, v_z

def velocidad_ion(temperature, mass):
    kb = 1.38064E-23  # Constante de Boltzmann en m^2 g s^-2 K^-1, kelvin, metros, segundos, kilogramos
    v = np.sqrt((2 * kb * temperature) / mass) * random.gauss(0, 1)
    return v

#Colision se detiene si átomo sale de segunda esfera de radio 2E-7
def event_condition(t, y, a, q, Omega, C4, C6, m_ion, m_a):
    # Definir condición de evento
    ionx, iony, ionz, ionvx, ionvy, ionvz, atomx, atomy, atomz, atomvx, atomvy, atomvz = y
    s = np.sqrt((atomx)**2 + (atomy)**2 + (atomz)**2)
    return s - 2.0E-6  #0.1E-6

# Indicar que el evento se debe buscar a través del cruce con cero
event_condition.terminal = True
event_condition.direction = 1

# Definición del vector de tiempo
t_span = (ti, tf)

# Condiciones iniciales y simulación
def run_simulation(i):
    ionx_0, iony_0, ionz_0 = 0.0, 0.0, 0.0
    ionvx_0 = velocidad_ion(T_ion, m_ion)
    ionvy_0 = velocidad_ion(T_ion, m_ion)
    ionvz_0 = velocidad_ion(T_ion, m_ion)

    atom1x_0, atom1y_0, atom1z_0 = position_on_sphere(r0)
    atom1vx_0, atom1vy_0, atom1vz_0 = velocidad_atom(T_atom, m_atom, atom1x_0, atom1y_0, atom1z_0)

    atom2x_0, atom2y_0, atom2z_0 = position_on_sphere(r0)
    atom2vx_0, atom2vy_0, atom2vz_0 = velocidad_atom(T_atom, m_atom, atom2x_0, atom2y_0, atom2z_0)

    y0 = [ionx_0, iony_0, ionz_0, ionvx_0, ionvy_0, ionvz_0,
          atom1x_0, atom1y_0, atom1z_0, atom1vx_0, atom1vy_0, atom1vz_0,
          atom2x_0, atom2y_0, atom2z_0, atom2vx_0, atom2vy_0, atom2vz_0]

    results = []

    for collision in range(num_collisions):
        solution = solve_ivp(system, (ti, tf), y0, t_eval=np.linspace(ti, tf, dt), method='RK45',
                             args=(a, q, Omega, C4, C6, m_ion, m_atom), rtol=ptol)

        results.append(solution)

        # Actualizar condiciones iniciales
        atom1x_0, atom1y_0, atom1z_0 = position_on_sphere(r0)
        atom1vx_0, atom1vy_0, atom1vz_0 = velocidad_atom(T_atom, m_atom, atom1x_0, atom1y_0, atom1z_0)

        atom2x_0, atom2y_0, atom2z_0 = position_on_sphere(r0)
        atom2vx_0, atom2vy_0, atom2vz_0 = velocidad_atom(T_atom, m_atom, atom2x_0, atom2y_0, atom2z_0)

        y0 = [ionx_0, iony_0, ionz_0, ionvx_0, ionvy_0, ionvz_0,
              atom1x_0, atom1y_0, atom1z_0, atom1vx_0, atom1vy_0, atom1vz_0,
              atom2x_0, atom2y_0, atom2z_0, atom2vx_0, atom2vy_0, atom2vz_0]

    return results

if __name__ == '__main__':
    with Pool(processes=num_cpus) as pool:
        all_results = pool.map(run_simulation, range(num_solution))

    radios = []
    tiempos = []
    salida_times_general = []
    salida_times_general_2 = []
    salida_times_general_3 = []
    # Contador para las resonancias formadas
    num_resonancias_formadas = 0
    num_resonancias_formadas_2 = 0
    num_resonancias_formadas_3 = 0
    num_three_body_recombination = 0
    for i, results in enumerate(all_results):   # Cada simulación completa
        salida_times = []
        salida_times_2 = []
        salida_times_3 = []
        for j, result in enumerate(results):    # Cada colisión dentro de una simulación
            # Extraer tiempos y soluciones
            t_sol = result.t  # Acceso a tiempos
            sol = result.y    # Acceso a soluciones

            # Extraer coordenadas y velocidades del ion y átomos
            ionx_sol, iony_sol, ionz_sol = sol[0], sol[1], sol[2]
            atom1x_sol, atom1y_sol, atom1z_sol = sol[6], sol[7], sol[8]
            atom2x_sol, atom2y_sol, atom2z_sol = sol[12], sol[13], sol[14]

            # Calcular el radio entre el átomo 1 y el ion
            r1_sol = np.sqrt((ionx_sol - atom1x_sol)**2 + (iony_sol - atom1y_sol)**2 + (ionz_sol - atom1z_sol)**2)
            r2_sol = np.sqrt((ionx_sol - atom2x_sol)**2 + (iony_sol - atom2y_sol)**2 + (ionz_sol - atom2z_sol)**2)
            r3_sol = np.sqrt((atom1x_sol - atom2x_sol)**2 + (atom1y_sol - atom2y_sol)**2 + (atom1z_sol - atom2z_sol)**2)

            # Graficar
            #fig, axs = plt.subplots(1, 2, figsize=(18, 6))
            #axs[0].plot(ionx_sol, iony_sol, label='Ion')
            #axs[0].plot(atom1x_sol, atom1y_sol, label='Átomo 1')
            #axs[0].plot(atom2x_sol, atom2y_sol, label='Átomo 2')
            #axs[0].legend()
            #axs[0].set_title('Posición en X-Y')
            #axs[0].set_xlim(-0.2e-6, 0.2e-6)
            #axs[0].set_ylim(-0.3e-6, 0.3e-6)


            #axs[1].plot(t_sol*1e6, r1_sol, label= 'ion-atom1')
            #axs[1].plot(t_sol*1e6, r2_sol, label= 'ion-atom2')
            #axs[1].plot(t_sol*1e6, r3_sol, label= 'atom1-atom2')
            #axs[1].legend()
            #axs[1].set_title('Radio vs Tiempo')
            #axs[1].set_xlabel('Tiempo (μs)')
            #axs[1].set_ylabel('Radio')
            #axs[1].set_xlim(0, 50)
            #axs[1].set_ylim(0, 0.5e-6)
            #plt.show()
            #num_resonancias_formadas = 0

            # Encontrar los índices donde los tres radios están por debajo del umbral
            indices_juntos = np.where((r3_sol < r_umbral))[0]
            if len(indices_juntos) >= 3:
                diferencias_3 = np.diff(indices_juntos)
                if np.any(diferencias_3 > 1):  # Si hay al menos un intervalo entre cruces
                    # Primer y último cruce por debajo del umbral
                    t_inicio = t_sol[indices_juntos[0]]  # Primer instante donde se cumplen las condiciones
                    t_fin = t_sol[indices_juntos[-1]]    # Último instante antes de que algún radio supere el umbral
                    tiempo_juntos = (t_fin - t_inicio)  # Convertir a μs
                    salida_times_3.append(tiempo_juntos  * 1E6)
                    num_resonancias_formadas_3 += 1
                    #print(f"Los cuerpos estuvieron juntos por {tiempo_juntos} μs")
                #else:
                    #print("Los cuerpos no estuvieron juntos).")


            cruces_abajo = np.where(r1_sol < umbral)[0]
                #Calculo de tiempo entre el primer radio minimo entre ion y atomo y el último
            if len(cruces_abajo) >= 4:
                # Calcular la diferencia entre cruces consecutivos para detectar oscilaciones
                diferencias = np.diff(cruces_abajo)
                if np.any(diferencias > 1):  # Si hay al menos un intervalo entre cruces
                    # Primer y último cruce por debajo del umbral
                    primer_cruce = t_sol[cruces_abajo[0]]
                    ultimo_cruce = t_sol[cruces_abajo[-1]]
                    tiempo_resonancia = ultimo_cruce - primer_cruce
                    salida_times.append(tiempo_resonancia * 1E6)  # Para cada run
                    num_resonancias_formadas += 1
                    if tiempo_resonancia > t_three_body:
                        num_three_body_recombination += 1
                        #print("Se presentó una three_body_recombination para el ion1-atom1")
                    #print(f"Resonancia ion-atom1 formada - Tiempo de resonancia: {tiempo_resonancia * 1E6:.2f} μs")
                #else:
                    #print("No se formó una resonancia ion-atom1 (oscilación no clara)")
            #else:
                #print("No se formó una resonancia ion-atom1 (menos de 4 cruces).")

            cruces_abajo_2 = np.where(r2_sol < umbral)[0]
            if len(cruces_abajo_2) >= 4:
                # Calcular la diferencia entre cruces consecutivos para detectar oscilaciones
                diferencias_2 = np.diff(cruces_abajo_2)
                if np.any(diferencias_2 > 1):  # Si hay al menos un intervalo entre cruces
                    # Primer y último cruce por debajo del umbral
                    primer_cruce_2 = t_sol[cruces_abajo_2[0]]
                    ultimo_cruce_2 = t_sol[cruces_abajo_2[-1]]
                    tiempo_resonancia_2 = ultimo_cruce_2 - primer_cruce_2
                    salida_times_2.append(tiempo_resonancia_2 * 1E6)  # Para cada run
                    num_resonancias_formadas_2 += 1
                    if tiempo_resonancia_2 > t_three_body:
                        num_three_body_recombination += 1
                        #print("Se presentó una three_body_recombination para el ion1-atom2")
                    #print(f"Resonancia ion-atom2 formada - Tiempo de resonancia: {tiempo_resonancia_2 * 1E6:.2f} μs")
                #else:
                    #print("No se formó una resonancia ion-atom2 (oscilación no clara)")
            #else:
                #print("No se formó una resonancia ion-atom2 (menos de 4 cruces).")

        # Calcular el promedio de tiempos de resonancia para esta simulación
        if salida_times:
            salida_times_average = np.mean(salida_times)
            salida_times_general.append(salida_times_average)
        if salida_times_2:
            salida_times_average_2 = np.mean(salida_times_2)
            salida_times_general_2.append(salida_times_average_2)
        if salida_times_3:
            salida_times_average_3 = np.mean(salida_times_3)
            salida_times_general_3.append(salida_times_average_3)

    # Calcular el promedio general de tiempos de resonancia

    salida_times_promedio = np.mean(salida_times_general)
    desv_tiempos = np.std(salida_times_general)
    #desv_tiempos = desv1
    salida_times_promedio_2 = np.mean(salida_times_general_2)
    desv_tiempos_2 = np.std(salida_times_general_2)
    salida_times_promedio_3 = np.mean(salida_times_general_3)
    desv_tiempos_3 = np.std(salida_times_general_3)

    num_coli = len(all_results) * len(results)
    tasa_resonancia = num_resonancias_formadas / num_coli
    desv_tasa = sqrt(num_resonancias_formadas * num_coli / (num_coli**3))
    tasa_resonancia_2 = num_resonancias_formadas_2 / num_coli
    desv_tasa_2 = sqrt(num_resonancias_formadas_2 * num_coli / (num_coli**3))
    tasa_resonancia_3 = num_resonancias_formadas_3 / num_coli
    desv_tasa_3 = sqrt(num_resonancias_formadas_3 * num_coli / (num_coli**3))
    tasa_three_body_recombination = num_three_body_recombination / num_coli
    desv_three_body_recombination = sqrt(num_three_body_recombination * num_coli / (num_coli**3))

    print(f'Tiempo de vida, tasa y desv = {salida_times_promedio}, {desv_tiempos}, {tasa_resonancia}, {desv_tasa}')
    print(f'Tiempo de vida2, tasa2 y desv2 = {salida_times_promedio_2}, {desv_tiempos_2}, {tasa_resonancia_2}, {desv_tasa_2}')
    print(f'Tiempo de vida tres cuerpos, tasa y desv = {salida_times_promedio_3}, {desv_tiempos_3}, {tasa_resonancia_3}, {desv_tasa_3}')
    print(f'Casos de three body: {num_three_body_recombination}')
    print(f'Tasa_three_body_recombination y desv (threebody) = {tasa_three_body_recombination}, {desv_three_body_recombination}')
