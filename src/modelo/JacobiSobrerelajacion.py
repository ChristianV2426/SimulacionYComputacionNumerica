"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""

import numpy as np


class JacobiSobrerelajacion:
	"""
	Esta clase contiene el algoritmo que aplica el método de Jacobi con sobrerelajación
	para resolver sistemas de ecuaciones lineales. 
	"""
	
	def __init__(self, matriz_coeficientes, matriz_constantes, omega, tolerancia):
		self.matriz_coeficientes = matriz_coeficientes.copy()
		self.matriz_constantes = matriz_constantes.copy()
		self.omega = omega
		self.tolerancia = tolerancia
		self.max_iteraciones = 1000

		# Verificar que la matriz de coeficientes sea cuadrada
		if (matriz_coeficientes.shape[0] == matriz_coeficientes.shape[1]):
			self.n = matriz_coeficientes.shape[0]
		else:
			raise Exception("JacobiSobrerelajacion: La matriz de coeficientes ingresada no es cuadrada.")
		
		# Verificar que la matriz de constantes tenga el mismo número de filas que la matriz de coeficientes
		if (self.n != matriz_constantes.shape[0]):
			raise Exception("JacobiSobrerelajacion: La matriz de coeficientes y la matriz de constantes no tienen el mismo número de filas.")	
		
		# Verificar que la matriz de coeficientes sea diagonal dominante
		if (self.matriz_coeficientes_diagonal_dominante()):
			print("JacobiSobrerelajacion: La matriz de coeficientes es diagonal dominante. El método de Jacobi convergerá utilizando cualquier estimación inicial.")
		else:
			print("JacobiSobrerelajacion: La matriz de coeficientes no es diagonal dominante. El método de Jacobi podría no converger.")
		
		print("JacobiSobrerelajacion: El número máximo de iteraciones en las que se intentará llegar a la solución es:", self.max_iteraciones)
			
		# Transformar las matrices de coeficientes y constantes para que las iteraciones tengan forma de punto fijo
		for i in range(self.n):
			a_ii = self.matriz_coeficientes[i][i]
			
			# Matriz de coeficientes
			for j in range(self.n):
				if (i != j):
					self.matriz_coeficientes[i][j] /= -a_ii
				else:
					self.matriz_coeficientes[i][j] = 0
			
			# Matriz de constantes
			self.matriz_constantes[i] /= a_ii

	
	def matriz_coeficientes_diagonal_dominante(self):
		for i in range(self.n):
			a_ii = abs(self.matriz_coeficientes[i][i])
			suma = 0

			for j in range(self.n):
				if (i != j):
					suma += abs(self.matriz_coeficientes[i][j])
			
			if (a_ii < suma):
				return False
			
		return True
	

	def resolver_sistema(self):
		solucion = np.zeros((self.n, 1))

		# La solución inicial será el vector de constantes, que previamente se transformó para que sus elementos
		# tuviera la forma b_i / a_ii
		solucion_previa = self.matriz_constantes.copy()

		# Si la solución previa inicial es cero, entonces se suma un valor muy pequeño a cada elemento del vector
		# para evitar errores de división por cero en la verificación de la tolerancia
		if (np.linalg.norm(solucion_previa) == 0):
			solucion_previa += 0.1

		for iteracion in range(self.max_iteraciones):
			for i in range(self.n):
				aporte_misma_variable = 0
				aporte_demas_variables = 0
				aporte_constante = self.matriz_constantes[i][0]

				for j in range(self.n):
					if (i == j):
						aporte_misma_variable = (1 - self.omega) * solucion_previa[i][0]
					else:
						aporte_demas_variables += self.matriz_coeficientes[i][j] * solucion_previa[j][0]
				
				solucion[i][0] = aporte_misma_variable + self.omega * (aporte_demas_variables + aporte_constante)
			
			# Verificar si la solución actual cumple con la tolerancia
			norma_solucion = np.linalg.norm(solucion)
			norma_diferencia = np.linalg.norm(solucion - solucion_previa)

			if ((norma_diferencia/norma_solucion) < self.tolerancia):
				break

			# Actualizar la solución previa
			solucion_previa = solucion.copy()
		
		# Verificar si se alcanzó la tolerancia antes de llegar al máximo número de iteraciones
		if (iteracion == self.max_iteraciones - 1):
			print("*JacobiSobrerelajacion: No se alcanzó la tolerancia en", self.max_iteraciones, "iteraciones.\n")
		else:
			print("*JacobiSobrerelajacion: Se alcanzó la tolerancia en", iteracion + 1, "iteraciones.\n")
		
		return solucion
