"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""


import numpy as np


class GradienteConjugado:
	"""
	Esta clase contiene el algoritmo que aplica el método del gradiente conjugado
	para resolver sistemas de ecuaciones lineales.
	"""

	def __init__(self, matriz_coeficientes, matriz_constantes, vector_valores_iniciales, tolerancia):
		self.matriz_coeficientes = matriz_coeficientes.copy()
		self.matriz_constantes = matriz_constantes.copy()
		self.vector_valores_iniciales = vector_valores_iniciales
		self.tolerancia = tolerancia
		self.max_iteraciones = 1000
	
		# Verificar que la matriz de coeficientes sea simétrica
		if (not self.matriz_coeficientes_simetrica()):
			print("GradienteConjugado: La matriz de coeficientes ingresada no es simétrica. El método no convergerá.")
		
		# Verificar que la matriz de coeficientes sea definida positiva
		if (not self.matriz_coeficientes_definida_positiva()):
			print("GradienteConjugado: La matriz de coeficientes ingresada no es definida positiva. El método no convergerá.")

		print("GradienteConjugado: El número máximo de iteraciones en las que se intentará llegar a la solución es:", self.max_iteraciones)		
	

	def matriz_coeficientes_simetrica(self):
		return np.array_equal(self.matriz_coeficientes, self.matriz_coeficientes.T)
	

	def matriz_coeficientes_definida_positiva(self):
		return np.all(np.linalg.eigvals(self.matriz_coeficientes) > 0)
	

	def resolver_sistema(self):
		solucion = np. zeros((self.vector_valores_iniciales.shape))
		solucion_anterior = self.vector_valores_iniciales.copy()

		# Si la solución inicial es cero, entonces se suma un valor muy pequeño a cada elemento del vector
		# para evitar errores de división por cero en las operaciones del ciclo iterativo
		if (np.all(solucion_anterior == 0)):
			solucion_anterior += 0.1

		r_anterior = self.matriz_constantes - np.matmul(self.matriz_coeficientes, solucion_anterior)
		d_anterior = r_anterior.copy()


		for iteracion in range(self.max_iteraciones):
			alpha = np.divide(np.matmul(r_anterior.T, r_anterior),  np.matmul(d_anterior.T, np.matmul(self.matriz_coeficientes, d_anterior)))

			solucion = solucion_anterior + (alpha * d_anterior)

			r = r_anterior - (alpha * np.matmul(self.matriz_coeficientes, d_anterior))
			norma_r = np.linalg.norm(r)

			# Verificar si la solución actual cumple con la tolerancia
			if (norma_r < self.tolerancia):
				break
				
			beta = np.divide(np.matmul(r.T, r), np.matmul(r_anterior.T, r_anterior))
			d = r + (beta * d_anterior)

			# Actualizar valores para la siguiente iteración
			solucion_anterior = solucion.copy()
			r_anterior = r.copy()
			d_anterior = d.copy()
		
		# Verificar si se llegó al número máximo de iteraciones
		if (iteracion == self.max_iteraciones - 1):
			print("GradienteConjugado: No se alcanzó la tolerancia en", self.max_iteraciones, "iteraciones.\n")
		else:
			print("GradienteConjugado: Se alcanzó la tolerancia en", iteracion + 1, "iteraciones.\n")
		
		return solucion
