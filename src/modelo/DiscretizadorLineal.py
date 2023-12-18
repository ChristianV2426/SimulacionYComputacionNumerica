"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""

import numpy as np


class DiscretizadorLineal:
	"""
	Esta clase contiene los métodos que permiten discretizar las ecuaciones diferenciales que
	describen el fenómeno. Para ello, se utiliza el método de diferencias finitas.
	Esta clase conoce el objeto frontera, que contiene toda la información relacionada con las
	condiciones de frontera del problema.

	Dado que esta clase pretende generar un sistema de ecuaciones lineales que describa el fenómeno,
	los métodos de esta clase reciben como parámetros las matrices de coeficientes y constantes
	de las velocidades inicializadas en ceros, y las modifican para que contengan los valores
	resultantes de aplicar la discretización lineal a cada punto de la malla.
	"""

	def __init__(self, frontera):
		self.frontera = frontera


	# Ecuación diferencial discretizada para vx:
	# -8vx(i, j) + vx(i+1, j) + 3vx(i-1, j) + vx(i, j+1) + 3vx(i, j-1) = 0
	def discretizar_velocidad_x(self, matriz_coeficientes, matriz_constantes):
		for i in range(1, 10):
			for j in range(1, 6):

				if not self.frontera.es_condicion_frontera(i, j):
					# Velocidad en el punto actual
					punto_actual = self.frontera.get_punto(i, j)
					matriz_coeficientes[punto_actual - 1][punto_actual - 1] = -8

					# Velocidad en el punto a la derecha
					if not self.frontera.es_condicion_frontera(i + 1, j):
						punto_derecha = self.frontera.get_punto(i + 1, j)
						matriz_coeficientes[punto_actual - 1][punto_derecha - 1] = 1
					else:
						velocidad_derecha = self.frontera.get_velocidad_x(i + 1, j)
						matriz_constantes[punto_actual - 1][0] -= velocidad_derecha

					# Velocidad en el punto a la izquierda
					if not self.frontera.es_condicion_frontera(i - 1, j):
						punto_izquierda = self.frontera.get_punto(i - 1, j)
						matriz_coeficientes[punto_actual - 1][punto_izquierda - 1] = 3
					else:
						velocidad_izquierda = self.frontera.get_velocidad_x(i - 1, j)
						matriz_constantes[punto_actual - 1][0] -= 3 * velocidad_izquierda
					
					# Velocidad en el punto de arriba
					if not self.frontera.es_condicion_frontera(i, j + 1):
						punto_arriba = self.frontera.get_punto(i, j + 1)
						matriz_coeficientes[punto_actual - 1][punto_arriba - 1] = 1
					else:
						velocidad_arriba = self.frontera.get_velocidad_x(i, j + 1)
						matriz_constantes[punto_actual - 1][0] -= velocidad_arriba

					# Velocidad en el punto de abajo
					if not self.frontera.es_condicion_frontera(i, j - 1):
						punto_abajo = self.frontera.get_punto(i, j - 1)
						matriz_coeficientes[punto_actual - 1][punto_abajo - 1] = 3
					else:
						velocidad_abajo = self.frontera.get_velocidad_x(i, j - 1)
						matriz_constantes[punto_actual - 1][0] -= 3 * velocidad_abajo
	

	# Ecuación diferencial discretizada para vy:
	# -8vy(i, j) + vy(i+1, j) + 3vy(i-1, j) + vy(i, j+1) + 3vy(i, j-1) = 0
	def discretizar_velocidad_y(self, matriz_coeficientes, matriz_constantes):
		for i in range(1, 10):
			for j in range(1, 6):

				if not self.frontera.es_condicion_frontera(i, j):
					# Velocidad en el punto actual
					punto_actual = self.frontera.get_punto(i, j)
					matriz_coeficientes[punto_actual - 1][punto_actual - 1] = -8

					# Velocidad en el punto a la derecha
					if not self.frontera.es_condicion_frontera(i + 1, j):
						punto_derecha = self.frontera.get_punto(i + 1, j)
						matriz_coeficientes[punto_actual - 1][punto_derecha - 1] = 1
					else:
						velocidad_derecha = self.frontera.get_velocidad_y(i + 1, j)
						matriz_constantes[punto_actual - 1][0] -= velocidad_derecha

					# Velocidad en el punto a la izquierda
					if not self.frontera.es_condicion_frontera(i - 1, j):
						punto_izquierda = self.frontera.get_punto(i - 1, j)
						matriz_coeficientes[punto_actual - 1][punto_izquierda - 1] = 3
					else:
						velocidad_izquierda = self.frontera.get_velocidad_y(i - 1, j)
						matriz_constantes[punto_actual - 1][0] -= 3 * velocidad_izquierda
					
					# Velocidad en el punto de arriba
					if not self.frontera.es_condicion_frontera(i, j + 1):
						punto_arriba = self.frontera.get_punto(i, j + 1)
						matriz_coeficientes[punto_actual - 1][punto_arriba - 1] = 1
					else:
						velocidad_arriba = self.frontera.get_velocidad_y(i, j + 1)
						matriz_constantes[punto_actual - 1][0] -= velocidad_arriba

					# Velocidad en el punto de abajo
					if not self.frontera.es_condicion_frontera(i, j - 1):
						punto_abajo = self.frontera.get_punto(i, j - 1)
						matriz_coeficientes[punto_actual - 1][punto_abajo - 1] = 3
					else:
						velocidad_abajo = self.frontera.get_velocidad_y(i, j - 1)
						matriz_constantes[punto_actual - 1][0] -= 3 * velocidad_abajo
