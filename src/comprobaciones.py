"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""

"""
Intención:

En este archivo hay ejemplos que prueban que los métodos de solución de sistemas de ecuaciones
implementados en el proyecto funcionan correctamente.
"""


from modelo import *
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from math import ceil
np.set_printoptions(threshold=np.inf, linewidth=1000)


# =========================== Comprobación del correcto funcionamiento de los algoritmos ===========================
# ===========================        procesando sistemas cuya solución es conocida       ===========================
print("\n########################## Comprobación del correcto funcionamiento de los algoritmos ##########################\n")


print("\n\n\n########################## Método de Jacobi con sobrerelajación ##########################\n")
print("Ejemplo 1:\n")
ejemplo1_a = np.array([[3.0, -0.1, -0.2], [0.1, 7.0, -0.3], [0.3, -0.2, 10.0]])
ejemplo1_b = np.array([[7.85], [-19.3], [71.4]])

jacobi_ejemplo1 = JacobiSobrerelajacion(ejemplo1_a, ejemplo1_b, 1.05, 0.00001)
solucion_ejemplo1 = jacobi_ejemplo1.resolver_sistema()
print("Solución con Jacobi:\n", np.round(solucion_ejemplo1, 2))
print("Solución algebraica:\n", np.round(np.matmul(np.linalg.inv(ejemplo1_a), ejemplo1_b), 2))


print("\n\nEjemplo 2:\n")
ejemplo2_a = np.array([[2.0, 1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 1.0, 0.0, 0.0], [0.0, 1.0, 2.0, 1.0, 0.0], [0.0, 0.0, 1.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0, 2.0]])
ejemplo2_b = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]])

jacobi_ejemplo2 = JacobiSobrerelajacion(ejemplo2_a, ejemplo2_b, 1.05, 0.00001)
solucion_ejemplo2 = jacobi_ejemplo2.resolver_sistema()
print("Solución con Jacobi:\n", np.round(solucion_ejemplo2, 2))
print("Solución algebraica:\n", np.round(np.matmul(np.linalg.inv(ejemplo2_a), ejemplo2_b), 2))



print("\n\n\n########################## Método del gradiente conjugado ##########################\n")
print("Ejemplo 3:\n")
ejemplo3_a = np.array([[2.0, 0.0], [0.0, 1.0]])
ejemplo3_b = np.array([[0.0], [0.0]])
ejemplo3_valores_iniciales = np.array([[0.0], [0.0]])

gradiente_conjugado_ejemplo3 = GradienteConjugado(ejemplo3_a, ejemplo3_b, ejemplo3_valores_iniciales, 0.00001)
solucion_ejemplo3 = gradiente_conjugado_ejemplo3.resolver_sistema()
print("Solución con gradiente conjugado:\n", np.round(solucion_ejemplo3, 2))
print("Solución algebraica:\n", np.round(np.matmul(np.linalg.inv(ejemplo3_a), ejemplo3_b), 2))


print("\n\nEjemplo 4:\n")
ejemplo4_a = np.array([[4, 2, 1], [2, 5, 0], [1, 0, 1]])
ejemplo4_b = np.array([[7], [8], [2]])
ejemplo4_valores_iniciales = np.array([[0.0], [0.0], [0.0]])

gradiente_conjugado_ejemplo4 = GradienteConjugado(ejemplo4_a, ejemplo4_b, ejemplo4_valores_iniciales, 0.00001)
solucion_ejemplo4 = gradiente_conjugado_ejemplo4.resolver_sistema()
print("Solución con gradiente conjugado:\n", np.round(solucion_ejemplo4, 2))
print("Solución algebraica:\n", np.round(np.matmul(np.linalg.inv(ejemplo4_a), ejemplo4_b), 2))



print("\n\n\n########################## Método de Newton-Raphson para sistemas no lineales ##########################\n")
print("Ejemplo 5:\n")
x, y  = sp.symbols('x y')
f1 = x**2 + x*y - 10
f2 = y + 3*x*y**2 - 57
ejemplo5_funciones = np.array([[f1], [f2]])
ejemplo5_variables = np.array([[x], [y]])
ejemplo5_valores_iniciales = np.array([[1.5], [3.5]])

newton_raphson_ejemplo5 = NewtonRaphson(ejemplo5_funciones, ejemplo5_variables, ejemplo5_valores_iniciales, 0.00001)
solucion_ejemplo5 = newton_raphson_ejemplo5.resolver_sistema()
print("Solución con Newton-Raphson:\n", np.round(solucion_ejemplo5, 2))
print("Comprobación: f(x) = 0\n", np.round(newton_raphson_ejemplo5.comprobar_solucion(), 2))


print("\n\nEjemplo 6:\n")
f3 = x**3 + y - 1
f4 = y**3 - x -1
ejemplo6_funciones = np.array([[f3], [f4]])
ejemplo6_variables = np.array([[x], [y]])
ejemplo6_valores_iniciales = np.array([[2.0], [-1.0]])

newton_raphson_ejemplo6 = NewtonRaphson(ejemplo6_funciones, ejemplo6_variables, ejemplo6_valores_iniciales, 0.00001)
solucion_ejemplo6 = newton_raphson_ejemplo6.resolver_sistema()
print("Solución con Newton-Raphson:\n", np.round(solucion_ejemplo6, 2))
print("Comprobación: f(x) = 0\n", np.round(newton_raphson_ejemplo6.comprobar_solucion(), 2))


print("\n\n\n########################## Comprobación visual de interpolación con splines bicúbicos ##########################\n")
matriz_aleatoria = np.random.rand(10, 10)
interpolador = Interpolador(matriz_aleatoria, 20)
matriz_interpolada = interpolador.interpolar()

figura, (grafico1, grafico2) = plt.subplots(1, 2)
mapa_calor1 = grafico1.pcolor(matriz_aleatoria, cmap="inferno")
mapa_calor2 = grafico2.pcolor(matriz_interpolada, cmap="inferno")

plt.colorbar(mapa_calor1, ax=grafico1, orientation="vertical", pad=0.05)
grafico1.set_title("Valores aleatorios", fontsize=12, fontweight="bold", pad=20)

plt.colorbar(mapa_calor2, ax=grafico2, orientation="vertical", pad=0.05)
grafico2.set_title("Valores interpolados", fontsize=12, fontweight="bold", pad=20)

# Ajuste de las marcas y etiquetas de los ejes del gráfico original
eje_x = np.arange(0, 10, 1)
eje_y = np.arange(0, 10, 1)
marcas_x = np.arange(0, 10, 1)
marcas_y = np.arange(0, 10, 1)

grafico1.set_xticks(marcas_x + 0.5, minor=False)
grafico1.set_yticks(marcas_y + 0.5, minor=False)
grafico1.set_xticklabels(eje_x, minor=False)
grafico1.set_yticklabels(eje_y, minor=False)

# Ajuste de las marcas y etiquetas de los ejes del gráfico interpolado
numero_filas, numero_columnas = matriz_interpolada.shape
marcas_x = np.arange(0, numero_columnas, 1)
marcas_y = np.arange(0, numero_filas, 1)

divisor_x = ceil(numero_columnas / 10)
divisor_y = ceil(numero_filas / 10)

eje_x = ['' if i % divisor_x != 0 else str(i / divisor_x) for i in range(numero_columnas)]
eje_y = ['' if i % divisor_y != 0 else str(i / divisor_y) for i in range(numero_filas)]

grafico2.set_xticks(marcas_x[::divisor_x] + 0.5, minor=False)
grafico2.set_yticks(marcas_y[::divisor_y] + 0.5, minor=False)
grafico2.set_xticklabels(eje_x[::divisor_x], minor=False)
grafico2.set_yticklabels(eje_y[::divisor_y], minor=False)

plt.show()
