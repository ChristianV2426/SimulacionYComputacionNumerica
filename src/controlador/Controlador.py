"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""

from modelo import *
import numpy as np
import matplotlib.pyplot as plt


class Controlador:
	"""
	Esta clase conoce todos los objetos que intervienen en la simulación, y se encarga de coordinar su ejecución
	según la interacción con el usuario desde la consola (dirigido desde el archivo main.py).
	"""

	def __init__(self, velocidad_x_inicial, omega, tolerancia):
		# Inicializar variables
		self.velocidad_x_inicial = velocidad_x_inicial
		self.omega = omega
		self.tolerancia = tolerancia

		# Inicializar objeto frontera y visualizador
		self.frontera = Frontera(self.velocidad_x_inicial)
		self.visualizador = Visualizador()

		# Inicializar objetos discretizadores
		self.discretizador_lineal = None
		self.discretizador_no_lineal = None

		# Inicializar soluciones
		# El programa calculará las soluciones una sola vez y llevará registro de ellas en las siguientes variables,
		# para así no tener que volver a calcular más de una vez una solución durante la interacción con el usuario.
		self.jacobi_vx = None
		self.jacobi_vy = None
		self.gradiente_conjugado_vx = None
		self.gradiente_conjugado_vy = None
		self.newton_raphson = None

		# Inicializar los objetos interpoladores
		self.interpolador_jacobi_vx = None
		self.interpolador_newton_raphson_vx = None

		# Inicializar matrices resumen de visualización
		self.vector_matrices_resumen_jacobi_vx = None
		self.vector_matrices_resumen_newton_raphson_vx = None


	def get_matriz_coeficientes_vx(self):
		if self.discretizador_lineal is None:
			self.discretizacion_lineal()
		return self.matriz_coeficientes_vx
	

	def get_matriz_constantes_vx(self):
		if self.discretizador_lineal is None:
			self.discretizacion_lineal()
		return self.matriz_constantes_vx
	

	def get_matriz_coeficientes_vy(self):
		if self.discretizador_lineal is None:
			self.discretizacion_lineal()
		return self.matriz_coeficientes_vy
	

	def get_matriz_constantes_vy(self):
		if self.discretizador_lineal is None:
			self.discretizacion_lineal()
		return self.matriz_constantes_vy
	

	def get_vector_funciones_no_lineales(self):
		if self.discretizador_no_lineal is None:
			self.discretizacion_no_lineal()
		return self.vector_funciones_no_lineales


	def get_vector_variables(self):
		if self.discretizador_no_lineal is None:
			self.discretizacion_no_lineal()
		return self.vector_variables
	

	def get_solucion_jacobi_vx(self):
		if self.jacobi_vx is None:
			self.solucion_jacobi()
		return self.solucion_jacobi_vx
	

	def get_comprobacion_jacobi_vx(self):
		if self.jacobi_vx is None:
			self.solucion_jacobi()
		return np.matmul(self.matriz_coeficientes_vx, self.solucion_jacobi_vx) - self.matriz_constantes_vx
	
	
	def get_matriz_visualizacion_vx_jacobi(self):
		if self.jacobi_vx is None:
			self.solucion_jacobi()
		return self.matriz_visualizacion_vx_jacobi
	

	def graficar_velocidad_x_jacobi(self):
		if self.jacobi_vx is None:
			self.solucion_jacobi()
		self.visualizador.graficar_velocidad_x(self.matriz_visualizacion_vx_jacobi, "Jacobi", None)
	

	def get_solucion_jacobi_vy(self):
		if self.jacobi_vy is None:
			self.solucion_jacobi()
		return self.solucion_jacobi_vy
	

	def get_comprobacion_jacobi_vy(self):
		if self.jacobi_vy is None:
			self.solucion_jacobi()
		return np.matmul(self.matriz_coeficientes_vy, self.solucion_jacobi_vy) - self.matriz_constantes_vy
	

	def get_matriz_visualizacion_vy_jacobi(self):
		if self.jacobi_vy is None:
			self.solucion_jacobi()
		return self.matriz_visualizacion_vy_jacobi


	def graficar_velocidad_y_jacobi(self):
		if self.jacobi_vy is None:
			self.solucion_jacobi()
		self.visualizador.graficar_velocidad_y(self.matriz_visualizacion_vy_jacobi, "Jacobi")

	
	def get_solucion_gradiente_conjugado_vx(self):
		if self.gradiente_conjugado_vx is None:
			self.solucion_gradiente_conjugado()
		return self.solucion_gradiente_conjugado_vx
	

	def get_comprobacion_gradiente_conjugado_vx(self):
		if self.gradiente_conjugado_vx is None:
			self.solucion_gradiente_conjugado()
		return np.matmul(self.matriz_coeficientes_vx, self.solucion_gradiente_conjugado_vx) - self.matriz_constantes_vx


	def get_matriz_visualizacion_vx_gradiente_conjugado(self):
		if self.gradiente_conjugado_vx is None:
			self.solucion_gradiente_conjugado()
		return self.matriz_visualizacion_vx_gradiente_conjugado


	def graficar_velocidad_x_gradiente_conjugado(self):
		if self.gradiente_conjugado_vx is None:
			self.solucion_gradiente_conjugado()
		self.visualizador.graficar_velocidad_x(self.matriz_visualizacion_vx_gradiente_conjugado, "Gradiente conjugado", None)


	def get_solucion_gradiente_conjugado_vy(self):
		if self.gradiente_conjugado_vy is None:
			self.solucion_gradiente_conjugado()
		return self.solucion_gradiente_conjugado_vy
	

	def get_comprobacion_gradiente_conjugado_vy(self):
		if self.gradiente_conjugado_vy is None:
			self.solucion_gradiente_conjugado()
		return np.matmul(self.matriz_coeficientes_vy, self.solucion_gradiente_conjugado_vy) - self.matriz_constantes_vy
	

	def get_matriz_visualizacion_vy_gradiente_conjugado(self):
		if self.gradiente_conjugado_vy is None:
			self.solucion_gradiente_conjugado()
		return self.matriz_visualizacion_vy_gradiente_conjugado
	

	def graficar_velocidad_y_gradiente_conjugado(self):
		if self.gradiente_conjugado_vy is None:
			self.solucion_gradiente_conjugado()
		self.visualizador.graficar_velocidad_y(self.matriz_visualizacion_vy_gradiente_conjugado, "Gradiente conjugado")


	def get_solucion_newton_raphson(self):
		if self.newton_raphson is None:
			self.solucion_newton_raphson()
		return self.solucion_newton_raphson
	

	def get_comprobacion_newton_raphson(self):
		if self.newton_raphson is None:
			self.solucion_newton_raphson()
		return self.newton_raphson.comprobar_solucion()
	

	def get_solucion_newton_raphson_vx(self):
		if self.newton_raphson is None:
			self.solucion_newton_raphson()
		return self.solucion_newton_raphson_vx
	

	def get_matriz_visualizacion_vx_newton_raphson(self):
		if self.newton_raphson is None:
			self.solucion_newton_raphson()
		return self.matriz_visualizacion_vx_newton_raphson


	def graficar_velocidad_x_newton_raphson(self):
		if self.newton_raphson is None:
			self.solucion_newton_raphson()
		self.visualizador.graficar_velocidad_x(self.matriz_visualizacion_vx_newton_raphson, "Newton-Raphson", None)
	

	def get_solucion_newton_raphson_vy(self):
		if self.newton_raphson is None:
			self.solucion_newton_raphson()
		return self.solucion_newton_raphson_vy
	

	def get_matriz_visualizacion_vy_newton_raphson(self):
		if self.newton_raphson is None:
			self.solucion_newton_raphson()
		return self.matriz_visualizacion_vy_newton_raphson
	

	def graficar_velocidad_y_newton_raphson(self):
		if self.newton_raphson is None:
			self.solucion_newton_raphson()
		self.visualizador.graficar_velocidad_y(self.matriz_visualizacion_vy_newton_raphson, "Newton-Raphson")


	def get_matriz_interpolacion_vx_jacobi(self, factor_aumento):
		self.interpolar_velocidad_x_jacobi(factor_aumento)
		return self.matriz_interpolacion_vx_jacobi


	def graficar_interpolacion_vx_jacobi(self, factor_aumento):
		self.interpolar_velocidad_x_jacobi(factor_aumento)
		print("Gráfica de la velocidad en x interpolada:\n")
		self.visualizador.graficar_velocidad_x(self.matriz_interpolacion_vx_jacobi, "Jacobi Interpolado", factor_aumento)


	def get_matriz_interpolacion_vx_newton_raphson(self, factor_aumento):
		self.interpolar_velocidad_x_newton_raphson(factor_aumento)
		return self.matriz_interpolacion_vx_newton_raphson
	

	def graficar_interpolacion_vx_newton_raphson(self, factor_aumento):
		self.interpolar_velocidad_x_newton_raphson(factor_aumento)
		print("Gráfica de la velocidad en x interpolada:\n")
		self.visualizador.graficar_velocidad_x(self.matriz_interpolacion_vx_newton_raphson, "Newton-Raphson Interpolado", factor_aumento)
	

	def mostrar_resumen_interpolacion_vx_jacobi(self):
		if self.vector_matrices_resumen_jacobi_vx is None:
			self.generar_matrices_resumen_jacobi()
		
		print("\nGráfico resumen:\n")
		self.visualizador.graficar_matrices_resumen(self.vector_matrices_resumen_jacobi_vx, "Jacobi")


	def mostrar_resumen_interpolacion_vx_newton_raphson(self):
		if self.vector_matrices_resumen_newton_raphson_vx is None:
			self.generar_matrices_resumen_newton_raphson()
		
		print("\nGráfico resumen:\n")
		self.visualizador.graficar_matrices_resumen(self.vector_matrices_resumen_newton_raphson_vx, "Newton-Raphson")


	def discretizacion_lineal(self):
		self.discretizador_lineal = DiscretizadorLineal(self.frontera)

		# Velocidad en x
		self.matriz_coeficientes_vx = np.zeros((33, 33))
		self.matriz_constantes_vx = np.zeros((33, 1))
		self.discretizador_lineal.discretizar_velocidad_x(self.matriz_coeficientes_vx, self.matriz_constantes_vx)

		# Velocidad en y
		self.matriz_coeficientes_vy = np.zeros((33, 33))
		self.matriz_constantes_vy = np.zeros((33, 1))
		self.discretizador_lineal.discretizar_velocidad_y(self.matriz_coeficientes_vy, self.matriz_constantes_vy)
	

	def discretizacion_no_lineal(self):
		self.discretizador_no_lineal = DiscretizadorNoLineal(self.frontera)
		
		self.vector_funciones_no_lineales = np.zeros((66, 1), dtype=object)
		self.discretizador_no_lineal.discretizar_velocidad(self.vector_funciones_no_lineales)
		self.vector_variables = self.discretizador_no_lineal.get_vector_variables()
	

	def solucion_jacobi(self):
		if self.discretizador_lineal is None:
			self.discretizacion_lineal()

		print("Velocidad en la componente x")
		# Calcular solución
		self.jacobi_vx = JacobiSobrerelajacion(self.matriz_coeficientes_vx, self.matriz_constantes_vx, self.omega, self.tolerancia)
		self.solucion_jacobi_vx = self.jacobi_vx.resolver_sistema()

		# Llenar matriz de visualización
		self.matriz_visualizacion_vx_jacobi = np.zeros((11, 7))
		self.frontera.llenar_matriz_visualizacion_vx(self.matriz_visualizacion_vx_jacobi, self.solucion_jacobi_vx)
		self.matriz_visualizacion_vx_jacobi = np.rot90(self.matriz_visualizacion_vx_jacobi, 1)
		self.matriz_visualizacion_vx_jacobi = np.flipud(self.matriz_visualizacion_vx_jacobi)

		print("\nVelocidad en la componente y")
		print("Para la solución vy el algoritmo dirá que no converge, pero esto es porque el vector solución es cero, por lo que hay problemas al momento de verificar la tolerancia (división por cero).", 
		" Sin embargo, tal como se observará en la comprobación, la solución es correcta.\n")
		# Calcular solución
		self.jacobi_vy = JacobiSobrerelajacion(self.matriz_coeficientes_vy, self.matriz_constantes_vy, self.omega, self.tolerancia)
		self.solucion_jacobi_vy = self.jacobi_vy.resolver_sistema()

		# Llenar matriz de visualización
		self.matriz_visualizacion_vy_jacobi = np.zeros((11, 7))
		self.frontera.llenar_matriz_visualizacion_vy(self.matriz_visualizacion_vy_jacobi, self.solucion_jacobi_vy)
		self.matriz_visualizacion_vy_jacobi = np.rot90(self.matriz_visualizacion_vy_jacobi, 1)
		self.matriz_visualizacion_vy_jacobi = np.flipud(self.matriz_visualizacion_vy_jacobi)
	

	def solucion_gradiente_conjugado(self):
		if self.discretizador_lineal is None:
			self.discretizacion_lineal()

		print("Velocidad en la componente x")
		# Calcular solución
		self.gradiente_conjugado_vx = GradienteConjugado(self.matriz_coeficientes_vx, self.matriz_constantes_vx, np.zeros((33, 1)), self.tolerancia)
		self.solucion_gradiente_conjugado_vx = self.gradiente_conjugado_vx.resolver_sistema()

		# Llenar matriz de visualización
		self.matriz_visualizacion_vx_gradiente_conjugado = np.zeros((11, 7))
		self.frontera.llenar_matriz_visualizacion_vx(self.matriz_visualizacion_vx_gradiente_conjugado, self.solucion_gradiente_conjugado_vx)
		self.matriz_visualizacion_vx_gradiente_conjugado = np.rot90(self.matriz_visualizacion_vx_gradiente_conjugado, 1)
		self.matriz_visualizacion_vx_gradiente_conjugado = np.flipud(self.matriz_visualizacion_vx_gradiente_conjugado)


		print("\nVelocidad en la componente y")
		# Calcular solución
		self.gradiente_conjugado_vy = GradienteConjugado(self.matriz_coeficientes_vy, self.matriz_constantes_vy, np.zeros((33, 1)), self.tolerancia)
		self.solucion_gradiente_conjugado_vy = self.gradiente_conjugado_vy.resolver_sistema()

		# Llenar matriz de visualización
		self.matriz_visualizacion_vy_gradiente_conjugado = np.zeros((11, 7))
		self.frontera.llenar_matriz_visualizacion_vy(self.matriz_visualizacion_vy_gradiente_conjugado, self.solucion_gradiente_conjugado_vy)
		self.matriz_visualizacion_vy_gradiente_conjugado = np.rot90(self.matriz_visualizacion_vy_gradiente_conjugado, 1)
		self.matriz_visualizacion_vy_gradiente_conjugado = np.flipud(self.matriz_visualizacion_vy_gradiente_conjugado)
	

	def solucion_newton_raphson(self):
		if self.discretizador_no_lineal is None:
			self.discretizacion_no_lineal()

		if self.jacobi_vx is None and self.jacobi_vy is None:
			print("Como solución inicial se utilizará la solución del método de Jacobi con sobrerelajación.")
			print("Como aun no se ha ejecutado el método de Jacobi con sobrerelajación, se ejecutará como paso previo.\n")

			self.solucion_jacobi()

			print("\nYa se tiene la solución inicial con el método de Jacobi. Se procede a ejecutar el método de Newton-Raphson.\n")
		
		self.solucion_inicial_newton_raphson = np.zeros((66, 1))
		self.solucion_inicial_newton_raphson[0:33, 0] =  self.solucion_jacobi_vx[:, 0]
		self.solucion_inicial_newton_raphson[33:66, 0] = self.solucion_jacobi_vy[:, 0]

		# Calcular solución
		self.newton_raphson = NewtonRaphson(self.vector_funciones_no_lineales, self.vector_variables, self.solucion_inicial_newton_raphson, self.tolerancia)
		self.solucion_newton_raphson = self.newton_raphson.resolver_sistema()
		self.solucion_newton_raphson_vx = self.solucion_newton_raphson[0:33, 0, np.newaxis].copy()
		self.solucion_newton_raphson_vy = self.solucion_newton_raphson[33:66, 0, np.newaxis].copy()

		# Llenar matriz de visualización para la velocidad en x
		self.matriz_visualizacion_vx_newton_raphson = np.zeros((11, 7))
		self.frontera.llenar_matriz_visualizacion_vx(self.matriz_visualizacion_vx_newton_raphson, self.solucion_newton_raphson_vx)
		self.matriz_visualizacion_vx_newton_raphson = np.rot90(self.matriz_visualizacion_vx_newton_raphson, 1)
		self.matriz_visualizacion_vx_newton_raphson = np.flipud(self.matriz_visualizacion_vx_newton_raphson)

		# Llenar matriz de visualización para la velocidad en y
		self.matriz_visualizacion_vy_newton_raphson = np.zeros((11, 7))
		self.frontera.llenar_matriz_visualizacion_vy(self.matriz_visualizacion_vy_newton_raphson, self.solucion_newton_raphson_vy)
		self.matriz_visualizacion_vy_newton_raphson = np.rot90(self.matriz_visualizacion_vy_newton_raphson, 1)
		self.matriz_visualizacion_vy_newton_raphson = np.flipud(self.matriz_visualizacion_vy_newton_raphson)


	def interpolar_velocidad_x_jacobi(self, factor_aumento):
		if self.jacobi_vx is None:
			print("Como aun no se ha ejecutado el método de Jacobi con sobrerelajación, se ejecutará como paso previo.\n")
			self.solucion_jacobi()
			print("\nYa se tiene la solución inicial con el método de Jacobi con sobrerelajación. Se procede a interpolar.\n")
		
		self.interpolador_jacobi_vx = Interpolador(self.matriz_visualizacion_vx_jacobi, factor_aumento)
		self.matriz_interpolacion_vx_jacobi = self.interpolador_jacobi_vx.interpolar()
	

	def interpolar_velocidad_x_newton_raphson(self, factor_aumento):
		if self.newton_raphson is None:
			print("Como aun no se ha ejecutado el método de Newton-Raphson, se ejecutará como paso previo.\n")
			self.solucion_newton_raphson()
			print("\nYa se tiene la solución inicial con el método de Newton-Raphson. Se procede a interpolar.\n")
		
		self.interpolador_newton_raphson_vx = Interpolador(self.matriz_visualizacion_vx_newton_raphson, factor_aumento)
		self.matriz_interpolacion_vx_newton_raphson = self.interpolador_newton_raphson_vx.interpolar()
	

	def generar_matrices_resumen_jacobi(self):
		if self.jacobi_vx is None:
			print("Como aun no se ha ejecutado el método de Jacobi con sobrerelajación, se ejecutará como paso previo.\n")
			self.solucion_jacobi()
			print("\nYa se tiene la solución inicial con el método de Jacobi con sobrerelajación. Se procede a generar las matrices resumen.\n")
		
		vector_factores = [2, 8, 32, 128]
		self.vector_matrices_resumen_jacobi_vx = []

		for factor in vector_factores:
			print("Calculando matriz interpolada con factor de aumento", factor, "...")
			interpolador = Interpolador(self.matriz_visualizacion_vx_jacobi, factor)
			self.vector_matrices_resumen_jacobi_vx.append(interpolador.interpolar())
	

	def generar_matrices_resumen_newton_raphson(self):
		if self.newton_raphson is None:
			print("Como aun no se ha ejecutado el método de Newton-Raphson, se ejecutará como paso previo.\n")
			self.solucion_newton_raphson()
			print("\nYa se tiene la solución inicial con el método de Newton-Raphson. Se procede a generar las matrices resumen.\n")
		
		vector_factores = [2, 8, 32, 128]
		self.vector_matrices_resumen_newton_raphson_vx = []

		for factor in vector_factores:
			print("Calculando matriz interpolada con factor de aumento", factor, "...")
			interpolador = Interpolador(self.matriz_visualizacion_vx_newton_raphson, factor)
			self.vector_matrices_resumen_newton_raphson_vx.append(interpolador.interpolar())	
