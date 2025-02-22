import numpy as np
from math import sqrt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count
from scipy.signal import find_peaks

#Ion: Yb+ , atom: Li

#constants
m_ion = 2.8733E-25  #(YB+) 2,8733965 × 10^-25 kg (SR+ 2.83E-25)
m_atom = 1.1525E-26 # (LITIO) 1,1525801 × 10^-26 kg (RB 1.44E-25 )
C4=5.607E-57    #C4=5.607E-57   Paper Prospects #C4=1.1E-56 RbSr+
C6=5E-19 * C4   #C6=5.0E-19     Paper Prospects (LIYB+) C6 = 2.803 × 10−75 Jm6 C4 = 2.803 × 10−57
frf = 2.5 * 10**6
Omega= 2 * np.pi * frf #MHz #Depende tambien mucho de esta fase
a = -2.982E-4  #-3E-7, da un buen comportamiento
q = 0.1      #Entre 0.1 y otro
#a = 0  #-3E-7, da un buen comportamiento
#q = 0     #Entre 0.1 y otro
T_ion =  0 # da un buen comportamiento  0.5 milikelvin   Paper Ziv Meir (de Jesus)
T_atom = 2E-6  #0.1E-6, da un buen comportamiento   #6 microkelvin (mucho más frio átomo que ion) Paper Ziv Meir
r0 = 0.15E-6
ti= 0
tf= 50E-6   #o 1.0E-5       #0.001, da un buen comportamiento pal ion #de matlab 200e-6;
dt= 1000
ptol = 1e-10
kB = 1.38064E-23
num_cpus = cpu_count()
num_solution = 1000
num_collisions = 1 #50 colisiones*3= 150 trayectorias que se estudian   500*10= 5000 trayectorias estudiadas    
umbral = 4E-8

def system(t, y, a, q, Omega, C4, C6, m_ion, m_a):

    ionx, iony, ionz, ionvx, ionvy, ionvz, atomx, atomy, atomz, atomvx, atomvy, atomvz = y

    # Definición de las ecuaciones diferenciales
    dydt = np.zeros_like(y)

    #Velocidades del ion
    dydt[0] = ionvx
    dydt[1] = ionvy
    dydt[2] = ionvz

    #velocidades del átomo
    dydt[6] = atomvx
    dydt[7] = atomvy
    dydt[8] = atomvz

    #Potencial para el ion
    dydt[3] = -(a+2*q*np.cos(Omega*t)) *ionx*(Omega**2 / 4)
    dydt[4] = -(a+2*q*np.cos(Omega*t)) *iony*(Omega**2 / 4)
    dydt[5] = -(2*a+2*-q*np.cos(Omega*t)) *ionz*(Omega**2 / 4)

    #Interacción con el potencial atom
    r = np.sqrt((atomx - ionx)**2 + (atomy - iony)**2 + (atomz - ionz)**2)

    dydt[3] += -(2*C4/r**5 -6 *C6 / r**7) * (ionx - atomx) / (m_ion * r)
    dydt[4] += -(2*C4/r**5 -6 *C6 / r**7) * (iony - atomy) / (m_ion * r)
    dydt[5] += -(2*C4/r**5 -6 *C6 / r**7) * (ionz - atomz) / (m_ion * r)

    dydt[9] += -(2*C4/r**5 -6 *C6 / r**7) * (atomx - ionx) / (m_atom * r)
    dydt[10] += -(2*C4/r**5 -6 *C6 / r**7) * (atomy - iony) / (m_atom * r)
    dydt[11] += -(2*C4/r**5 -6 *C6 / r**7) * (atomz - ionz) / (m_atom * r)

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



# Resolución del sistema de ecuaciones diferenciales
def run_simulation(i):

    ionx_0=0.0
    iony_0=0.0
    ionz_0=0.0
    ionvx_0=velocidad_ion(T_ion, m_ion)
    ionvy_0=velocidad_ion(T_ion, m_ion)
    ionvz_0=velocidad_ion(T_ion, m_ion)
    atomx_0, atomy_0, atomz_0 = position_on_sphere(r0)
    atomvx_0, atomvy_0, atomvz_0 = velocidad_atom(T_atom, m_atom, atomx_0, atomy_0, atomz_0)
    y0 = [ionx_0, iony_0, ionz_0, ionvx_0, ionvy_0, ionvz_0, atomx_0, atomy_0, atomz_0, atomvx_0, atomvy_0, atomvz_0]

    results = []

    for collision in range (num_collisions):
          #inicialization variables
        solution = solve_ivp(system, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], dt), method='RK45', args=(a, q, Omega, C4, C6, m_ion, m_atom), rtol=ptol)

        # Extracción de las soluciones temporales
        t_sol = solution.t
        ionx_sol = solution.y[0]
        iony_sol = solution.y[1]
        ionz_sol = solution.y[2]
        ionvx_sol = solution.y[3]
        ionvy_sol = solution.y[4]
        ionvz_sol = solution.y[5]
        atomx_sol = solution.y[6]
        atomy_sol = solution.y[7]
        atomz_sol = solution.y[8]
        atomvx_sol = solution.y[9]
        atomvy_sol = solution.y[10]
        atomvz_sol = solution.y[11]

        results.append([t_sol, ionx_sol, iony_sol, ionz_sol, ionvx_sol, ionvy_sol, ionvz_sol, atomx_sol, atomy_sol, atomz_sol, atomvx_sol, atomvy_sol, atomvz_sol])

        #Se actulizan las condiciones para el ion y el atomo
        atomx_0, atomy_0, atomz_0 = position_on_sphere(r0)
        atomvx_0, atomvy_0, atomvz_0 = velocidad_atom(T_atom, m_atom, atomx_0, atomy_0, atomz_0)
        #y0 = [ionx_sol[-1], iony_sol[-1], ionz_sol[-1], ionvx_sol[-1], ionvy_sol[-1], ionvz_sol[-1], atomx_0, atomy_0, atomz_0, atomvx_0, atomvy_0, atomvz_0]
        y0 = [ionx_0, iony_0, ionz_0, ionvx_0, ionvy_0, ionvz_0, atomx_0, atomy_0, atomz_0, atomvx_0, atomvy_0, atomvz_0]

    return results

if __name__ == '__main__':
    with Pool(processes=num_cpus) as pool:
        all_results = pool.map(run_simulation, range(num_solution))
    #Ekin_total = []
            #E_kin = 0.5 * m_ion * (ionvx_sol**2 + ionvy_sol**2 + ionvz_sol**2)
            #Tkin = np.mean(0.5 * m_ion * (ionvx_sol[1:]**2 + ionvy_sol[1:]**2 + ionvz_sol[1:]) * (t_sol[1:] - t_sol[:-1]) / (t_sol[-1] - t_sol[0]) * (len(t_sol) - 1)) * 1 / (3/2 * kB) * 1e6
            #Ekin_colision.append(Tkin)
        #Ekin_total.append(Ekin_colision)

    #Ekin_total_promedio = np.mean(Ekin_total, axis=0)
    #print("Ekin_total_promedio", Ekin_total_promedio)


    #plt.plot(Ekin_total_promedio, label='Average Kinetic Energy')
    #plt.title('Average Kinetic Energy as a Function of Number of Collisions')
    #plt.xlabel('Number of collisions')
    #plt.ylabel('Average kinetic energy (μK)')
    #plt.grid()
    #plt.show()
    # Listas para almacenar los radios y los tiempos
    umbral = 4e-8
    radios = []
    tiempos = []
    salida_times_general = []

    # Contador para las resonancias formadas
    num_resonancias_formadas = 0

    for i, results in enumerate(all_results):   # Cada simulación completa
        salida_times = []
        for j, result in enumerate(results):    # Cada colisión dentro de una simulación
            # Extraer tiempos y soluciones
            t_sol = result[0]
            ionx_sol = result[1]
            iony_sol = result[2]
            ionz_sol = result[3]
            atomx_sol = result[7]
            atomy_sol = result[8]
            atomz_sol = result[9]

            # Calcular el radio entre el átomo y el ion en cada instante de tiempo
            r_sol = np.sqrt((ionx_sol - atomx_sol)**2 + (iony_sol - atomy_sol)**2 + (ionz_sol - atomz_sol)**2)
            radios.append(r_sol)
            tiempos.append(t_sol)

            # Crear la figura con subplots
            #fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            # Primer gráfico: Posición del ion y átomo en X-Y
            #axs[0].plot(ionx_sol, iony_sol, label='Ion')
            #axs[0].plot(atomx_sol, atomy_sol, label='Átomo')
            #axs[0].set_xlim(-2E-7, 2E-7)
            #axs[0].set_ylim(-1E-6, 1E-6)
            #axs[0].set_xlabel('X')
            #axs[0].set_ylabel('Y')
            #axs[0].set_title('Posición en X-Y del Ion y Átomo')
            #axs[0].legend()
            #axs[0].grid(True)

            # Segundo gráfico: Radio vs tiempo
            #axs[1].plot(t_sol * 1E6, r_sol * 1E6, label=f'Colisión {j + 1}')
            #axs[1].set_xlabel('Tiempo [microsegundos]')
            #axs[1].set_ylabel('Radio [micrómetros]')
            #axs[1].set_title('Radio vs Tiempo')
            #axs[1].legend()
            #axs[1].set_xlim(0, 50)
            #axs[1].set_ylim(0, 0.2)
            #axs[1].grid(True)

            # Tercer gráfico: Texto con información de resonancia
            #axs[2].axis('off')  # Desactivar ejes para el tercer subplot


            cruces_abajo = np.where(r_sol < umbral)[0]  # Índices donde el radio está por debajo del umbral
            # Verificar si hay al menos 4 cruces y si hay una oscilación clara

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
                    resonancia_texto = f"Resonancia formada\nTiempo de resonancia: {tiempo_resonancia * 1E6:.2f} μs"
                    #print(f"Resonancia formada\nTiempo de resonancia: {tiempo_resonancia * 1E6:.2f} μs")
                else:
                    resonancia_texto = "No se formó una resonancia (oscilación no clara)."
                    #print("No seвается una resonancia (oscilación no clara).")
            else:
                resonancia_texto = "No se formó una resonancia (menos de 4 cruces)."
                #print("No seвается una resonancia (menos de 4 cruces).")

            # Agregar texto al tercer subplot
            #axs[2].text(0.5, 0.5, resonancia_texto, fontsize=12, ha='center', va='center')

            # Mostrar la figura
            #plt.tight_layout()
            #plt.show()

        # Calcular el promedio de tiempos de resonancia para esta simulación
        if salida_times:
            salida_times_average = np.mean(salida_times)
            salida_times_general.append(salida_times_average)
            desv1 = np.std(salida_times)

    # Calcular el promedio general de tiempos de resonancia
    if salida_times_general:
        salida_times_promedio = np.mean(salida_times_general)
        desv_tiempos = np.std(salida_times_general)
        #desv_tiempos = desv1

        num_coli = len(all_results) * len(results)
        tasa_resonancia = num_resonancias_formadas / num_coli
        desv_tasa = sqrt(num_resonancias_formadas * num_coli / (num_coli**3))
        print(f'Tiempo de vida, tasa y desv = {salida_times_promedio}, {desv_tiempos}, {tasa_resonancia}, {desv_tasa}')
    else:
        print("No se formaron resonancias en ninguna colisión.")


