import numpy as np
from math import pi
import numpy.linalg as la

data = np.loadtxt("vis_depth/sup_coordinates.txt")
width = 544
height = data.shape[0] // width
xyz = data.reshape((height, width, 3))

p1 = (206, 290) # apple red Sup
p2 = (353, 290)

pt1 = xyz[p1[0], p1[1], :]
pt2 = xyz[p2[0], p2[1], :]

diam = la.norm(pt1 - pt2)
r = diam / 2.0
V = (4.0 / 3.0) * pi * (r**3)
massa_estim = V * 0.85

print(f"Raio estimado: {r:.3f} cm")
print(f"Volume estimado: {V:.2f} cm³")
print(f"Massa estimada: {massa_estim:.2f} g")
print("-" * 60)
massa_real = 175.0
densidade = 0.85
V_real = massa_real / densidade
scale = (V_real / V) ** (1/3)

print(f"Massa Real: {massa_real:.2f} g")

print(f"\nFator de calibração: {scale:.3f}")

r_corrigido = r * scale
V_corrigido = (4/3) * pi * (r_corrigido**3)
massa_corrigida = V_corrigido * densidade

print(f"Raio corrigido: {r_corrigido:.3f} cm")
print(f"Volume corrigido: {V_corrigido:.2f} cm³")
