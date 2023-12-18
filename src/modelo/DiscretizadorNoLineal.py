"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""

import numpy as np
import sympy as sp


class DiscretizadorNoLineal:
	"""
	Esta clase contiene los métodos que permiten discretizar las ecuaciones diferenciales que
	describen el fenómeno. Para ello, se utiliza el método de diferencias finitas.
	Esta clase conoce el objeto frontera, que contiene toda la información relacionada con las
	condiciones de frontera del problema.

	Dado que esta clase pretende generar un sistema de ecuaciones no lineales que describa el fenómeno,
	los métodos de esta clase devolverán matrices y arreglos de funciones, utilizando la representación
	que ofrece la librería sympy.
	"""

	# Constantes de la clase
	
	# Variables de velocidad en la componente x
	vx1, vx2, vx3, vx4, vx5, vx6, vx7, vx8, vx9, vx10 = sp.symbols("vx1 vx2 vx3 vx4 vx5 vx6 vx7 vx8 vx9 vx10")
	vx11, vx12, vx13, vx14, vx15, vx16, vx17, vx18, vx19, vx20 = sp.symbols("vx11 vx12 vx13 vx14 vx15 vx16 vx17 vx18 vx19 vx20")	
	vx21, vx22, vx23, vx24, vx25, vx26, vx27, vx28, vx29, vx30 = sp.symbols("vx21 vx22 vx23 vx24 vx25 vx26 vx27 vx28 vx29 vx30")
	vx31, vx32, vx33 = sp.symbols("vx31 vx32 vx33")

	# Variables de velocidad en la componente y
	vy1, vy2, vy3, vy4, vy5, vy6, vy7, vy8, vy9, vy10 = sp.symbols("vy1 vy2 vy3 vy4 vy5 vy6 vy7 vy8 vy9 vy10")
	vy11, vy12, vy13, vy14, vy15, vy16, vy17, vy18, vy19, vy20 = sp.symbols("vy11 vy12 vy13 vy14 vy15 vy16 vy17 vy18 vy19 vy20")
	vy21, vy22, vy23, vy24, vy25, vy26, vy27, vy28, vy29, vy30 = sp.symbols("vy21 vy22 vy23 vy24 vy25 vy26 vy27 vy28 vy29 vy30")
	vy31, vy32, vy33 = sp.symbols("vy31 vy32 vy33")


	def __init__(self, frontera):
		self.frontera = frontera
		self.vector_variables = np.array(
		[[self.vx1], [self.vx2], [self.vx3], [self.vx4], [self.vx5], [self.vx6], [self.vx7], [self.vx8], [self.vx9], [self.vx10],
		[self.vx11], [self.vx12], [self.vx13], [self.vx14], [self.vx15], [self.vx16], [self.vx17], [self.vx18], [self.vx19], [self.vx20],
		[self.vx21], [self.vx22], [self.vx23], [self.vx24], [self.vx25], [self.vx26], [self.vx27], [self.vx28], [self.vx29], [self.vx30],
		[self.vx31], [self.vx32], [self.vx33],
		[self.vy1], [self.vy2], [self.vy3], [self.vy4], [self.vy5], [self.vy6], [self.vy7], [self.vy8], [self.vy9], [self.vy10],
		[self.vy11], [self.vy12], [self.vy13], [self.vy14], [self.vy15], [self.vy16], [self.vy17], [self.vy18], [self.vy19], [self.vy20],
		[self.vy21], [self.vy22], [self.vy23], [self.vy24], [self.vy25], [self.vy26], [self.vy27], [self.vy28], [self.vy29], [self.vy30],
		[self.vy31], [self.vy32], [self.vy33]]
		)


	def discretizar_velocidad(self, vector_funciones):
		
		# indice que llevará registro de la fila del vector de funciones que se está llenando
		numero_funcion = 0
		
		for j in range(1, 6):
			for i in range(1, 10):

				if not self.frontera.es_condicion_frontera(i, j):

					# Punto actual
					punto_actual = self.frontera.get_punto(i, j)
					vx_punto_actual = self.vector_variables[punto_actual - 1][0]
					vy_punto_actual = self.vector_variables[punto_actual + 32][0]

					# Punto a la derecha
					if not self.frontera.es_condicion_frontera(i + 1, j):
						punto_derecha = self.frontera.get_punto(i + 1, j)
						vx_punto_derecha = self.vector_variables[punto_derecha - 1][0]
						vy_punto_derecha = self.vector_variables[punto_derecha + 32][0]
					else:
						vx_punto_derecha = self.frontera.get_velocidad_x(i + 1, j)
						vy_punto_derecha = self.frontera.get_velocidad_y(i + 1, j)
					
					# Punto a la izquierda
					if not self.frontera.es_condicion_frontera(i - 1, j):
						punto_izquierda = self.frontera.get_punto(i - 1, j)
						vx_punto_izquierda = self.vector_variables[punto_izquierda - 1][0]
						vy_punto_izquierda = self.vector_variables[punto_izquierda + 32][0]
					else:
						vx_punto_izquierda = self.frontera.get_velocidad_x(i - 1, j)
						vy_punto_izquierda = self.frontera.get_velocidad_y(i - 1, j)
					
					# Punto de arriba
					if not self.frontera.es_condicion_frontera(i, j + 1):
						punto_arriba = self.frontera.get_punto(i, j + 1)
						vx_punto_arriba = self.vector_variables[punto_arriba - 1][0]
						vy_punto_arriba = self.vector_variables[punto_arriba + 32][0]
					else:
						vx_punto_arriba = self.frontera.get_velocidad_x(i, j + 1)
						vy_punto_arriba = self.frontera.get_velocidad_y(i, j + 1)
					
					# Punto de abajo
					if not self.frontera.es_condicion_frontera(i, j - 1):
						punto_abajo = self.frontera.get_punto(i, j - 1)
						vx_punto_abajo = self.vector_variables[punto_abajo - 1][0]
						vy_punto_abajo = self.vector_variables[punto_abajo + 32][0]
					else:
						vx_punto_abajo = self.frontera.get_velocidad_x(i, j - 1)
						vy_punto_abajo = self.frontera.get_velocidad_y(i, j - 1)
					
					# Llenar el vector de funciones

					# Ecuación diferencial discretizada para vx:
					# -8vx(i, j) + 2vx(i+1, j) + 2vx(i-1, j) + 2vx(i, j+1) + 2vx(i, j-1)
					# -vx(i, j)*vx(i+1, j) + vx(i, j)*vx(i-1, j) -vy(i, j)*vx(i, j+1) + vy(i, j)*vx(i, j-1) = 0

					vector_funciones[numero_funcion][0] = (
					- (8 * vx_punto_actual) + (2 * vx_punto_derecha) + (2 * vx_punto_izquierda) + (2 * vx_punto_arriba) + (2 * vx_punto_abajo) 
					- (vx_punto_actual * vx_punto_derecha) + (vx_punto_actual * vx_punto_izquierda) - (vy_punto_actual * vx_punto_arriba) + (vy_punto_actual * vx_punto_abajo)
					)

					# Ecuación diferencial discretizada para vy:
					# -8vy(i, j) + 2vy(i+1, j) + 2vy(i-1, j) + 2vy(i, j+1) + 2vy(i, j-1)
					# -vx(i, j)*vy(i+1, j) + vx(i, j)*vy(i-1, j) -vy(i, j)*vy(i, j+1) + vy(i, j)*vy(i, j-1) = 0
					vector_funciones[numero_funcion + 33][0] = (
					- (8 * vy_punto_actual) + (2 * vy_punto_derecha) + (2 * vy_punto_izquierda) + (2 * vy_punto_arriba) + (2 * vy_punto_abajo)
					- (vx_punto_actual * vy_punto_derecha) + (vx_punto_actual * vy_punto_izquierda) - (vy_punto_actual * vy_punto_arriba) + (vy_punto_actual * vy_punto_abajo)
					)

					# Incrementar el indice de registro del vector de funciones
					numero_funcion += 1
	
	def get_vector_variables(self):
		return self.vector_variables
