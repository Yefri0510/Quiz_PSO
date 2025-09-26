# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 08:54:09 2025

@author: yefri
"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Definir parámetros del problema
area_size = 5000  # metros
grid_resolution = 100  # metros por celda, para discretizar
grid_points = int(area_size / grid_resolution) + 1  # 51x51 grid (0 to 5000 en pasos de 100)
x = np.linspace(0, area_size, grid_points)
y = np.linspace(0, area_size, grid_points)
X, Y = np.meshgrid(x, y)

# Crear un mapa de probabilidades simulado (como no se proporciona, usamos gaussianas para zonas calientes)
# Supongamos 3 zonas con alta probabilidad
centers = np.array([[1000, 1000], [3000, 2000], [4000, 4000]])
probs = np.zeros_like(X)
for center in centers:
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    probs += 0.5 * np.exp(-dist**2 / (2 * 1000**2))  # Gaussianas con sigma=1000m
probs = np.clip(probs, 0, 1)  # Probabilidades entre 0 y 1

# Aplanar el grid para cálculos
grid_coords = np.c_[X.ravel(), Y.ravel()]
cell_area = grid_resolution ** 2  # Área por celda, pero como probs son densidades, suma probs (asumiendo densidad)

# Parámetros
num_drones = 10
detection_radius = 200  # metros
max_time = 120  # minutos
drone_speed = 10  # m/s, asumidos por los 36 km/h típicos
max_time_seconds = max_time * 60  # 7200 segundos
max_distance = drone_speed * max_time_seconds

# Posición inicial de drones, asumida en el centro
initial_pos = np.array([2500, 2500])

# Función de fitness: maximizar suma de probs cubiertas, penalizar si tiempo excede
def fitness(particle):
    # particle: (num_drones * 2) dimensiones, x1,y1,x2,y2,...
    positions = particle.reshape(num_drones, 2)
    
    # Calcular tiempo: max distancia desde initial a cada pos / speed
    dists = np.linalg.norm(positions - initial_pos, axis=1)
    times = dists / drone_speed  # segundos
    max_t = np.max(times) / 60  # minutos
    if max_t > max_time:
        return -np.inf  # Penalizar si excede tiempo
    
    # Cobertura: para cada punto en grid, si dist a algún dron <= radius, cubrir
    dist_to_drones = cdist(grid_coords, positions)
    covered = np.any(dist_to_drones <= detection_radius, axis=1)
    covered_probs = probs.ravel()[covered].sum()
    
    # Objetivo: maximizar covered_probs, minimizar tiempo
    return covered_probs - 0.1 * max_t  # Pequeña penalización por tiempo

# Parámetros PSO
num_particles = 50
dimensions = num_drones * 2
max_iter = 100
w = 0.5  # inercia
c1 = 1.5  # cognitivo
c2 = 1.5  # social

# Inicializar partículas
particles = np.random.uniform(0, area_size, (num_particles, dimensions))
velocities = np.random.uniform(-100, 100, (num_particles, dimensions))  # velocidades iniciales

pbest = particles.copy()
pbest_fitness = np.array([fitness(p) for p in particles])
gbest_idx = np.argmax(pbest_fitness)
gbest = particles[gbest_idx].copy()
gbest_fitness = pbest_fitness[gbest_idx]

# Iteraciones PSO
for iter in range(max_iter):
    for i in range(num_particles):
        r1 = np.random.rand(dimensions)
        r2 = np.random.rand(dimensions)
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest[i] - particles[i]) +
                         c2 * r2 * (gbest - particles[i]))
        
        particles[i] += velocities[i]
        # Clampear posiciones
        particles[i] = np.clip(particles[i], 0, area_size)
        
        fit = fitness(particles[i])
        if fit > pbest_fitness[i]:
            pbest[i] = particles[i].copy()
            pbest_fitness[i] = fit
            if fit > gbest_fitness:
                gbest = particles[i].copy()
                gbest_fitness = fit

# Resultados
best_positions = gbest.reshape(num_drones, 2)
best_prob = gbest_fitness + 0.1 * (np.max(np.linalg.norm(best_positions - initial_pos, axis=1)) / drone_speed / 60)  # Quitar penalización para prob real
best_time = np.max(np.linalg.norm(best_positions - initial_pos, axis=1)) / drone_speed / 60

print("Mejores posiciones de drones:")
print(best_positions)
print(f"Probabilidad cubierta maximizada: {best_prob}")
print(f"Tiempo mínimo requerido: {best_time} minutos")
print(f"Fitness final: {gbest_fitness}")

# Generar gráfico
fig, ax = plt.subplots(figsize=(10, 10))
# Mapa de probabilidades como heatmap
im = ax.imshow(probs, extent=[0, area_size, 0, area_size], origin='lower', cmap='hot', alpha=0.7)
plt.colorbar(im, ax=ax, label='Probabilidad')

# Posiciones iniciales
ax.plot(initial_pos[0], initial_pos[1], 'go', markersize=10, label='Posición inicial')

# Posiciones de drones
ax.scatter(best_positions[:, 0], best_positions[:, 1], c='b', marker='o', label='Drones')

# Radios de detección
for pos in best_positions:
    circle = plt.Circle(pos, detection_radius, color='b', fill=False, linestyle='--')
    ax.add_artist(circle)

ax.set_title('Mapa de Probabilidades con Posiciones Óptimas de Drones')
ax.set_xlabel('X (metros)')
ax.set_ylabel('Y (metros)')
ax.legend()
plt.show()