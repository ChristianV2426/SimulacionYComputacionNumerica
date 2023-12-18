"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""

import numpy as np


class Frontera:
	"""
	Esta clase contiene datos y métodos relacionados con las condiciones de frontera del probelema.

	La velocidad del fluido en la frontera se guarda en diccionarios, donde la clave es la tupla (x, y)
	que representa las coordenadas del punto de interés en la malla resultante de la discretización.

	Esta clase contiene métodos que permiten consultar la velocidad en un punto de la frontera, así como
	como métodos que permiten consultar si un punto está o no en la frontera.
	"""

	# Constantes de la clase 
	condiciones_frontera_velocidad_x = {
		# Borde inferior, donde y = 0
		(1, 0) : 0,
		(2, 0) : 0, 
		(3, 0) : 0,
		(4, 0) : 0,
		(5, 0) : 0,
		(6, 0) : 0,
		(7, 0) : 0,
		(8, 0) : 0,
		(9, 0) : 0,
		(10, 0) : 0,

		# Borde superior, donde y = 6
		(1, 6) : 0,
		(2, 6) : 0,
		(3, 6) : 0,
		(4, 6) : 0,
		(5, 6) : 0,
		(6, 6) : 0,
		(7, 6) : 0,
		(8, 6) : 0,
		(9, 6) : 0,
		(10, 6) : 0,

		# Borde derecho, donde x = 10
		(10, 1) : 0,
		(10, 2) : 0,
		(10, 3) : 0,
		(10, 4) : 0,
		(10, 5) : 0,
		(10, 6) : 0,

		# Viga inferior
		(4, 1) : 0,
		(4, 2) : 0,
		(5, 2) : 0,
		(6, 2) : 0,
		(6, 1) : 0,
		(5, 1) : 0,

		# Viga superior
		(4, 5) : 0,
		(4, 4) : 0,
		(5, 4) : 0,
		(6, 4) : 0,
		(6, 5) : 0,
		(5, 5) : 0
	}

	condiciones_frontera_velocidad_y = {
		# Borde inferior, donde y = 0
		(0, 0) : 0,
		(1, 0) : 0,
		(2, 0) : 0,
		(3, 0) : 0,
		(4, 0) : 0,
		(5, 0) : 0,
		(6, 0) : 0,
		(7, 0) : 0,
		(8, 0) : 0,
		(9, 0) : 0,
		(10, 0) : 0,

		#Borde izquierdo, donde x = 0
		(0, 1) : 0,
		(0, 2) : 0,
		(0, 3) : 0,
		(0, 4) : 0,
		(0, 5) : 0,

		# Borde superior, donde y = 6
		(0, 6) : 0,
		(1, 6) : 0,
		(2, 6) : 0,
		(3, 6) : 0,
		(4, 6) : 0,
		(5, 6) : 0,
		(6, 6) : 0,
		(7, 6) : 0,
		(8, 6) : 0,
		(9, 6) : 0,
		(10, 6) : 0,

		# Borde derecho, donde x = 10
		(10, 1) : 0,
		(10, 2) : 0,
		(10, 3) : 0,
		(10, 4) : 0,
		(10, 5) : 0,

		# Viga inferior
		(4, 1) : 0,
		(4, 2) : 0,
		(5, 2) : 0,
		(6, 2) : 0,
		(6, 1) : 0,
		(5, 1) : 0,
		
		# Viga superior
		(4, 5) : 0,
		(4, 4) : 0,
		(5, 4) : 0,
		(6, 4) : 0,
		(6, 5) : 0,
		(5, 5) : 0
	}

	de_coordenadas_a_punto = {
		(1, 1) : 1,
		(2, 1) : 2,
		(3, 1) : 3,
		(7, 1) : 4,
		(8, 1) : 5,
		(9, 1) : 6,

		(1, 2) : 7,
		(2, 2) : 8,
		(3, 2) : 9,
		(7, 2) : 10,
		(8, 2) : 11,
		(9, 2) : 12,

		(1, 3) : 13,
		(2, 3) : 14,
		(3, 3) : 15,
		(4, 3) : 16,
		(5, 3) : 17,
		(6, 3) : 18,
		(7, 3) : 19,
		(8, 3) : 20,
		(9, 3) : 21,

		(1, 4) : 22,
		(2, 4) : 23,
		(3, 4) : 24,
		(7, 4) : 25,
		(8, 4) : 26,
		(9, 4) : 27,

		(1, 5) : 28,
		(2, 5) : 29,
		(3, 5) : 30,
		(7, 5) : 31,
		(8, 5) : 32,
		(9, 5) : 33
	}


	def __init__(self, velocidad_x_inicial):
		self.velocidad_x_inicial = velocidad_x_inicial
		self.inicializar_velocidad_x()

	
	def inicializar_velocidad_x(self):
		self.condiciones_frontera_velocidad_x.update({
			# Borde izquierdo, donde x = 0
            (0, 0) : self.velocidad_x_inicial,
			(0, 1) : self.velocidad_x_inicial,
			(0, 2) : self.velocidad_x_inicial,
			(0, 3) : self.velocidad_x_inicial,
			(0, 4) : self.velocidad_x_inicial,
			(0, 5) : self.velocidad_x_inicial,
			(0, 6) : self.velocidad_x_inicial
		})
	

	def get_condiciones_frontera_velocidad_x(self):
		return self.condiciones_frontera_velocidad_x
	

	def get_velocidad_x(self, i, j):
		return self.condiciones_frontera_velocidad_x[(i, j)]
	

	def get_condiciones_frontera_velocidad_y(self):
		return self.condiciones_frontera_velocidad_y
	

	def get_velocidad_y(self, i, j):
		return self.condiciones_frontera_velocidad_y[(i, j)]
		

	def get_de_coordenadas_a_punto(self):
		return self.de_coordenadas_a_punto
	

	def get_punto(self, i, j):
		return self.de_coordenadas_a_punto[(i, j)]
		

	def get_velocidad_x_inicial(self):
		return self.velocidad_x_inicial
	

	def es_condicion_frontera(self, i, j):
		return (i, j) in self.condiciones_frontera_velocidad_x or (i, j) in self.condiciones_frontera_velocidad_y
	
	
	def llenar_matriz_visualizacion_vx(self, matriz_visualizacion, solucion):
		for i in range(11):
			for j in range(7):
				
				# Si el punto está en la frontera, se guarda la velocidad conocida de la frontera
				if (self.es_condicion_frontera(i, j)):
					matriz_visualizacion[i][j] = self.get_velocidad_x(i, j)
				
				# Si el punto no está en la frontera, se guarda la velocidad calculada en el vector solución
				else:
					matriz_visualizacion[i][j] = solucion[self.get_punto(i, j) - 1][0]


	def llenar_matriz_visualizacion_vy(self, matriz_visualizacion, solucion):
		for i in range(11):
			for j in range(7):
				
				# Si el punto está en la frontera, se guarda la velocidad conocida de la frontera
				if (self.es_condicion_frontera(i, j)):
					matriz_visualizacion[i][j] = self.get_velocidad_y(i, j)
				
				# Si el punto no está en la frontera, se guarda la velocidad calculada en el vector solución
				else:
					matriz_visualizacion[i][j] = solucion[self.get_punto(i, j) - 1][0]