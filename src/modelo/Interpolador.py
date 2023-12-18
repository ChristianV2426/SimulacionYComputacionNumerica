"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""

import numpy as np


class Interpolador:
	"""
	Esta clase contiene datos y métodos que permiten interpolar los puntos de la malla en los que no se conoce
	el valor de la velocidad del fluido.

	La técnica de interpolación utilizada son los splines bicúbicos.
	"""

	def __init__(self, malla_velocidades, factor_aumento):
		self.malla_velocidades = malla_velocidades
		self.factor_aumento = factor_aumento
		
		self.matriz_coeficientes_a = np.array([
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
			[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
			[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
			[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0],
			[0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9] 
			])
		
		self.matriz_inversa_coeficientes_a = np.linalg.inv(self.matriz_coeficientes_a)

		self.matriz_coeficientes_derivada = np.array([
			[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 2, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 2, 0, 0, 0, 0],
			[0, -2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
			[0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
			[0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 2, 0],
			[1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0],
			[0, 1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 1, 0],
			[0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 1]
			])
		

	def valor_imagen(self, i, j):
		numero_filas, numero_columnas = self.malla_velocidades.shape
		
		if (i == -1):
			return self.valor_imagen(0, j)
		
		elif (i == numero_filas):
			return self.valor_imagen(numero_filas - 1, j)
		
		elif (j == -1):
			return self.valor_imagen(i, 0)
		
		elif (j == numero_columnas):
			return self.valor_imagen(i, numero_columnas - 1)
		
		else:
			return self.malla_velocidades[i][j]


	def interpolar(self):
		# Ajustar el tamaño de la matriz interpolada y crear el vector de imágenes
		filas, columnas = self.malla_velocidades.shape
		matriz_interpolada = np.zeros((filas + (filas - 1) * self.factor_aumento, columnas + (columnas - 1) * self.factor_aumento))
		vector_imagenes = np.zeros((16, 1))

		for fila in range(filas - 1):
			for columna in range(columnas - 1):
				
				# Llenar el vector de imágenes con los valores de las velocidades en los puntos de la malla
				# en y alrededor del área a interpolar
				vector_imagenes[0][0] = self.valor_imagen(fila - 1, columna - 1)
				vector_imagenes[1][0] = self.valor_imagen(fila - 1, columna)
				vector_imagenes[2][0] = self.valor_imagen(fila - 1, columna + 1)
				vector_imagenes[3][0] = self.valor_imagen(fila - 1, columna + 2)
				vector_imagenes[4][0] = self.valor_imagen(fila, columna - 1)
				vector_imagenes[5][0] = self.valor_imagen(fila, columna)
				vector_imagenes[6][0] = self.valor_imagen(fila, columna + 1)
				vector_imagenes[7][0] = self.valor_imagen(fila, columna + 2)
				vector_imagenes[8][0] = self.valor_imagen(fila + 1, columna - 1)
				vector_imagenes[9][0] = self.valor_imagen(fila + 1, columna)
				vector_imagenes[10][0] = self.valor_imagen(fila + 1, columna + 1)
				vector_imagenes[11][0] = self.valor_imagen(fila + 1, columna + 2)
				vector_imagenes[12][0] = self.valor_imagen(fila + 2, columna - 1)
				vector_imagenes[13][0] = self.valor_imagen(fila + 2, columna)
				vector_imagenes[14][0] = self.valor_imagen(fila + 2, columna + 1)
				vector_imagenes[15][0] = self.valor_imagen(fila + 2, columna + 2)

				# Calcular los coeficientes del polinomio interpolador
				vector_coeficientes = np.matmul(self.matriz_inversa_coeficientes_a, (1/4 * np.matmul(self.matriz_coeficientes_derivada, vector_imagenes)))

				# Interpolar los puntos de la malla dentro del área de interés usando el polinomio interpolador calculado
				for j in range(self.factor_aumento + 2):
					for i in range(self.factor_aumento + 2):

						coordenada_x = (1/(self.factor_aumento + 1)) * i
						coordenada_y = (1/(self.factor_aumento + 1)) * j

						matriz_interpolada[fila * (self.factor_aumento + 1) + j][columna * (self.factor_aumento + 1) + i] = (
							vector_coeficientes[0] + 
							vector_coeficientes[1] * coordenada_x +
							vector_coeficientes[2] * coordenada_x**2 +
							vector_coeficientes[3] * coordenada_x**3 +
							vector_coeficientes[4] * coordenada_y +
							vector_coeficientes[5] * coordenada_x * coordenada_y +
							vector_coeficientes[6] * coordenada_x**2 * coordenada_y +
							vector_coeficientes[7] * coordenada_x**3 * coordenada_y +
							vector_coeficientes[8] * coordenada_y**2 +
							vector_coeficientes[9] * coordenada_x * coordenada_y**2 +
							vector_coeficientes[10] * coordenada_x**2 * coordenada_y**2 +
							vector_coeficientes[11] * coordenada_x**3 * coordenada_y**2 +
							vector_coeficientes[12] * coordenada_y**3 +
							vector_coeficientes[13] * coordenada_x * coordenada_y**3 +
							vector_coeficientes[14] * coordenada_x**2 * coordenada_y**3 +
							vector_coeficientes[15] * coordenada_x**3 * coordenada_y**3
						)

		return matriz_interpolada
