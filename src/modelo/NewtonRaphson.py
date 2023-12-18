"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""

import numpy as np
import sympy as sp


class NewtonRaphson:
	"""
	Esta clase contiene el algoritmo que aplica el método de Newton-Raphson
	para resolver sistemas de ecuaciones no lineales.
	"""
		
	def __init__(self, vector_funciones, vector_variables, vector_valores_iniciales, tolerancia):
		self.vector_funciones = vector_funciones
		self.vector_variables = vector_variables
		self.vector_valores_iniciales = vector_valores_iniciales
		self.tolerancia = tolerancia
		
		self.numero_funciones = vector_funciones.shape[0]
		self.numero_variables = vector_variables.shape[0]
		self.max_iteraciones = 100

		print("Newton-Rapshon: El número máximo de iteraciones en las que se intentará llegar a la solución es:", self.max_iteraciones)
		
		# Calcular el jacobiano a partir del vector de funciones y el total de variables del sistema
		self.jacobiano = self.calcular_jacobiano()

	
	def calcular_jacobiano(self):
		jacobiano = np.zeros((self.numero_funciones, self.numero_variables), dtype=object)

		for i in range(self.numero_funciones):
			for j in range(self.numero_variables):
				jacobiano[i][j] = sp.diff(self.vector_funciones[i][0], self.vector_variables[j][0])

		return jacobiano


	def evaluar_matriz_funciones(self, matriz_funciones, vector_valores):
		tupla_variables = tuple(self.vector_variables)
		tupla_valores = tuple(vector_valores)

		numero_filas = matriz_funciones.shape[0]
		numero_columnas = matriz_funciones.shape[1]

		matriz_funciones_evaluadas = np.zeros((numero_filas, numero_columnas))

		for i in range(numero_filas):
			for j in range(numero_columnas):
				matriz_funciones_evaluadas[i][j] = sp.lambdify(tupla_variables, matriz_funciones[i][j])(*tupla_valores)
		
		return matriz_funciones_evaluadas
	

	def resolver_sistema(self):
		solucion = np.zeros((self.numero_variables, 1))

		# La solución inicial es el vector de valores iniciales propuesto al instanciar el objeto de esta clase
		solucion_previa = self.vector_valores_iniciales.copy()

		for iteracion in range(self.max_iteraciones):
			print("Newton-Rapshon: Iteracion:", iteracion + 1, "...")

			# Evaluar el vector de funciones con la solución previa
			funciones_evaluadas = self.evaluar_matriz_funciones(self.vector_funciones, solucion_previa)

			# Evaluar la matriz jacobiana con la solución previa
			jacobiano_evaluado = self.evaluar_matriz_funciones(self.jacobiano, solucion_previa)

			# Calcular la nueva iteración
			solucion = solucion_previa - np.matmul(np.linalg.inv(jacobiano_evaluado), funciones_evaluadas)

			# Verificar si la solución actual cumple con la tolerancia
			norma_solucion = np.linalg.norm(solucion)
			norma_diferencia = np.linalg.norm(solucion - solucion_previa)

			if ((norma_diferencia/norma_solucion) < self.tolerancia):
				break
			
			# Actualizar la solución previa
			solucion_previa = solucion.copy()

			# print("Solución:\n", np.round(solucion, 1))

		# Verificar si se alcanzó la tolerancia antes de llegar al máximo número de iteraciones
		if (iteracion == self.max_iteraciones - 1):
			print("*Newton-Rapshon: No se alcanzó la tolerancia en", self.max_iteraciones, "iteraciones.\n")
		else:
			print("*Newton-Rapshon: Se alcanzó la tolerancia en", iteracion + 1, "iteraciones.\n")
		
		self.solucion = solucion
		
		return solucion


	def comprobar_solucion(self):
		if (self.solucion is None):
			print("Newton-Rapshon: No se ha resuelto el sistema de ecuaciones no lineales. Primero debe llamar al método resolver_sistema() desde el objeto instanciado de esta clase.\n")
			return
		else:
			return self.evaluar_matriz_funciones(self.vector_funciones, self.solucion)
