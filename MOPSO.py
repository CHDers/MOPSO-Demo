'''
🚀🚀🚀🚀🚀🚀: 
Descripttion: Yanjun Hao的代码
version: 1.0.0
Author: Yanjun Hao
Date: 2023-11-26 12:19:06
LastEditors: Yanjun Hao
LastEditTime: 2023-11-26 12:46:44
'''
import numpy as np
from rich import print
from tqdm import tqdm
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])
        self.velocity = np.random.uniform(low=-1, high=1, size=bounds.shape[0])
        self.best_position = np.copy(self.position)
        self.best_objectives = objectives(self.position)

    def update(self, global_best):
        w = 0.5
        c1 = 0.8
        c2 = 0.9

        r1 = np.random.random(size=self.velocity.shape)
        r2 = np.random.random(size=self.velocity.shape)

        self.velocity = w * self.velocity + c1 * r1 * (self.best_position - self.position) + c2 * r2 * (global_best.position - self.position)
        self.position = self.position + self.velocity
        np.clip(self.position, bounds[:, 0], bounds[:, 1], out=self.position)

def objectives(x):
    f1 = x[0]**2 + x[1]**2
    f2 = (x[0]-1)**2 + x[1]**2
    return np.array([f1, f2])

def dominates(p, q):
    return all(p <= q) and any(p < q)

def update_archive(archive, new_particle):
    non_dominated = True
    updated_archive = []

    for particle in archive:
        if dominates(new_particle.best_objectives, particle.best_objectives):
            non_dominated = False
            break
        elif not dominates(particle.best_objectives, new_particle.best_objectives):
            updated_archive.append(particle)

    if non_dominated:
        updated_archive.append(new_particle)

    return updated_archive

def mopso(num_particles, bounds, num_iterations):
    swarm = [Particle(bounds) for _ in range(num_particles)]
    archive = []

    for _ in tqdm(range(num_iterations)):
        for particle in swarm:
            particle.update(np.random.choice(swarm))
            objectives_values = objectives(particle.position)
            if dominates(objectives_values, particle.best_objectives):
                particle.best_position = particle.position
                particle.best_objectives = objectives_values
                archive = update_archive(archive, particle)

    return archive

# 参数设置
num_particles = 50
bounds = np.array([[-10, 10], [-10, 10]])
num_iterations = 100

# 运行MOPSO
archive = mopso(num_particles, bounds, num_iterations)
print(len(archive))

# 绘制Pareto前沿
pareto_points = np.array([p.best_objectives for p in archive])
plt.scatter(pareto_points[:, 0], pareto_points[:, 1])
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Front')
plt.savefig("Pareto Front.svg", bbox_inches='tight')
plt.show()
