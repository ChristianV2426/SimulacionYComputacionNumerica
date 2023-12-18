"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


class Visualizador:
	"""
	Esta clase contiene métodos que permiten visualizar los resultados de la simulación.
	Se encarga de mostrar los mapas de calor de las velocidades en x e y, según el método
	de solución que haya sido utilizado.
	"""

	def __init__(self):
		self.valores_eje_x = np.arange(0, 11, 1)
		self.valores_eje_y = np.arange(0, 7, 1)


	def graficar_velocidad_x(self, matriz_visualizacion, metodo, aumento):
		numero_filas, numero_columnas = matriz_visualizacion.shape
	
		# Creación del mapa de calor
		figura, grafico = plt.subplots()
		mapa_calor = grafico.pcolor(matriz_visualizacion, cmap="PuBu")
		plt.colorbar(mapa_calor, ax=grafico, label="Velocidad en componente x", orientation="vertical", pad=0.05)
		
		# Ajuste de las marcas y etiquetas de los ejes
		marcas_x = np.arange(0, numero_columnas, 1)
		marcas_y = np.arange(0, numero_filas, 1)
		
		divisor_x = ceil(numero_columnas / 11)
		divisor_y = ceil(numero_filas / 7)

		eje_x = ['' if i % divisor_x != 0 else str(i / divisor_x) for i in range(numero_columnas)]
		eje_y = ['' if i % divisor_y != 0 else str(i / divisor_y) for i in range(numero_filas)]

		grafico.set_xticks(marcas_x[::divisor_x] + 0.5, minor=False)
		grafico.set_yticks(marcas_y[::divisor_y] + 0.5, minor=False)
		grafico.set_xticklabels(eje_x[::divisor_x], minor=False)
		grafico.set_yticklabels(eje_y[::divisor_y], minor=False)
		
		# Ajuste del título del gráfico
		if (metodo == "Jacobi"):
			grafico.set_title("Método de Jacobi - Velocidad en x", fontsize=12, fontweight="bold", pad=20)

		elif (metodo == "Jacobi Interpolado"):
			grafico.set_title("Interpolación a partir del método de Jacobi (aumento: " + str(aumento) + ")\nVelocidad en x", fontsize=12, fontweight="bold", pad=20)
		
		elif (metodo == "Gradiente conjugado"):
			grafico.set_title("Método del Gradiente Conjugado - Velocidad en x", fontsize=12, fontweight="bold", pad=20)

		elif (metodo == "Newton-Raphson"):
			grafico.set_title("Método de Newton-Raphson - Velocidad en x", fontsize=12, fontweight="bold", pad=20)

		elif (metodo == "Newton-Raphson Interpolado"):
			grafico.set_title("Interpolación a partir del método de Newton-Raphson (aumento: " + str(aumento) + ")\nVelocidad en x", fontsize=12, fontweight="bold", pad=20)
		
		plt.show()
	

	def graficar_velocidad_y(self, matriz_visualizacion, metodo):
		# Creación del mapa de calor
		figura, grafico = plt.subplots()
		mapa_calor = grafico.pcolor(matriz_visualizacion, cmap="Blues")
		plt.colorbar(mapa_calor, ax=grafico, label="Velocidad en componente y", orientation="vertical", pad=0.05)

		# Ajuste de las marcas y etiquetas de los ejes
		grafico.set_xticks(np.arange(self.valores_eje_x.shape[0]) + 0.5, minor=False)
		grafico.set_yticks(np.arange(self.valores_eje_y.shape[0]) + 0.5, minor=False)
		grafico.set_xticklabels(self.valores_eje_x, minor=False)
		grafico.set_yticklabels(self.valores_eje_y, minor=False)

		# Ajuste del título del gráfico
		if (metodo == "Jacobi"):
			grafico.set_title("Método de Jacobi - Velocidad en y", fontsize=12, fontweight="bold", pad=20)

		elif (metodo == "Gradiente conjugado"):
			grafico.set_title("Método del Gradiente Conjugado - Velocidad en y", fontsize=12, fontweight="bold", pad=20)

		elif (metodo == "Newton-Raphson"):
			grafico.set_title("Método de Newton-Raphson - Velocidad en y", fontsize=12, fontweight="bold", pad=20)
		
		plt.show()
	

	def graficar_matrices_resumen(self, vector_matrices_resumen, metodo):
		assert len(vector_matrices_resumen) == 4
		
		figura, graficos = plt.subplots(2, 2)
		
		if (metodo == "Jacobi"):
			figura.suptitle("Interpolación a partir del método Jacobi", fontsize=14, fontweight="bold")

		elif (metodo == "Newton-Raphson"):
			figura.suptitle("Interpolación a partir del método de Newton-Raphson", fontsize=14, fontweight="bold")
		
		for grafico, matriz_resumen, factor in zip(graficos.flat, vector_matrices_resumen, [2, 8, 32, 128]):
			mapa_calor = grafico.pcolor(matriz_resumen, cmap="Blues")
			plt.colorbar(mapa_calor, ax=grafico, label="Velocidad en componente x", orientation="vertical", pad=0.05)
			grafico.set_title("Aumento: " + str(factor), fontsize=12, fontweight="bold", pad=5)

			# Ajuste de las marcas y etiquetas de los ejes
			numero_filas, numero_columnas = matriz_resumen.shape
			marcas_x = np.arange(0, numero_columnas, 1)
			marcas_y = np.arange(0, numero_filas, 1)

			divisor_x = ceil(numero_columnas / 11)
			divisor_y = ceil(numero_filas / 7)

			eje_x = ['' if i % divisor_x != 0 else str(i / divisor_x) for i in range(numero_columnas)]
			eje_y = ['' if i % divisor_y != 0 else str(i / divisor_y) for i in range(numero_filas)]

			grafico.set_xticks(marcas_x[::divisor_x] + 0.5, minor=False)
			grafico.set_yticks(marcas_y[::divisor_y] + 0.5, minor=False)
			grafico.set_xticklabels(eje_x[::divisor_x], minor=False)
			grafico.set_yticklabels(eje_y[::divisor_y], minor=False)
	
		plt.show()

		
	
	

