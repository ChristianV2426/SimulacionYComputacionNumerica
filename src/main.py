"""
	Simulación y Computación Numérica - 750016C Grupo 01
	Proyecto del curso -
	Simulación numérica de flujo incompresible utilizando las ecuaciones de Navier-Stokes

	Autor:
	Christian David Vargas Gutiérrez - 2179172
"""


from controlador import Controlador
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=1000)


if __name__ == "__main__":
	velocidad_x_inicial = 5
	omega = 1.05
	tolerancia = 0.0001

	controlador = Controlador(velocidad_x_inicial, omega, tolerancia)
	
	# Interacción con el usuario
	while(True):
		print("\n\n########################## Menú principal ##########################")
		print("Ingrese la opción que desea ejecutar:\n")

		print("1. Generación del sistema de ecuaciones lineales")
		print("2. Generación del sistema de ecuaciones no lineales")
		print("3. Solución del sistema de ecuaciones lineales utilizando el método de Jacobi con sobrerelajación")
		print("4. Solución del sistema de ecuaciones lineales utilizando el método del gradiente conjugado")
		print("5. Solución del sistema de ecuaciones no lineales utilizando el método de Newton-Raphson")
		print("6. Interpolación individual de resultados obtenidos con el método de Jacobi con sobrerelajación")
		print("7. Gráfico resumen interpolación de resultados obtenidos con el método de Jacobi con sobrerelajación")
		print("8. Interpolación individual de resultados obtenidos con el método de Newton-Raphson")
		print("9. Gráfico resumen interpolación de resultados obtenidos con el método de Newton-Raphson")
		print("10. Salir")

		opcion = input("\nOpción: ")

		if opcion == "1":
			print("\n\n########################## Discretización de las ecuaciones diferenciales, variante sistema de ecuaciones lineales ##########################")
			
			print("\nVelocidad en la componente x.\nSistema de la forma A vx = B:")
			print("\nMatriz (A) de coeficientes de la velocidad en x\n", controlador.get_matriz_coeficientes_vx())
			print("\nMatriz (B) de constantes de la velocidad en x\n", controlador.get_matriz_constantes_vx())

			print("\n\nVelocidad en la componente y.\nSistema de la forma A vy = B:")
			print("\nMatriz (A) de coeficientes de la velocidad en y\n", controlador.get_matriz_coeficientes_vy())
			print("\nMatriz (B) de constantes de la velocidad en y\n",controlador.get_matriz_constantes_vy())
		

		elif opcion == "2":
			print("\n\n########################## Discretización de las ecuaciones diferenciales, variante sistema de ecuaciones no lineales ##########################")
			print("\nVector de funciones no lineales\n", controlador.get_vector_funciones_no_lineales())
		
		
		elif opcion == "3":
			print("\n\n########################## Método de Jacobi con sobrerelajación ##########################\n")

			# Mostrar resultados para la velocidad en x
			print("Vector solución vx:\n", np.round(controlador.get_solucion_jacobi_vx(), 1))
			print("\nComprobación: A vx - B = 0\n", np.round(controlador.get_comprobacion_jacobi_vx(), 2))
			print("Gráfica de la velocidad en x:\n")
			controlador.graficar_velocidad_x_jacobi()

			# Mostrar resultados para la velocidad en y
			print("\n\nVector solución vy:\n", np.round(controlador.get_solucion_jacobi_vy(), 1))
			print("\nComprobación: A vy - B = 0\n", np.round(controlador.get_comprobacion_jacobi_vy(), 2))
			print("Gráfica de la velocidad en y:\n")
			controlador.graficar_velocidad_y_jacobi()


		elif opcion == "4":
			print("\n\n########################## Método del gradiente conjugado ##########################\n")
						
			# Mostrar resultados para la velocidad en x
			print("Vector solución vx:\n", np.round(controlador.get_solucion_gradiente_conjugado_vx(), 1))
			print("\nComprobación: A vx - B = 0\n", np.round(controlador.get_comprobacion_gradiente_conjugado_vx(), 0))
			print("Gráfica de la velocidad en x:\n")
			controlador.graficar_velocidad_x_gradiente_conjugado()

			# Mostrar resultados para la velocidad en y
			print("\n\nVector solución vy:\n", np.round(controlador.get_solucion_gradiente_conjugado_vy(), 1))
			print("\nComprobación: A vy - B = 0\n", np.round(controlador.get_comprobacion_gradiente_conjugado_vy(), 1))
			print("Gráfica de la velocidad en y:\n")
			controlador.graficar_velocidad_y_gradiente_conjugado()
		

		elif opcion == "5":
			print("\n\n########################## Método de Newton-Raphson para sistemas no lineales ##########################\n")
			
			# Mostrar resultados para el sistema único de velocidades tanto en x como en y
			print("Vector solución:\n", np.round(controlador.get_solucion_newton_raphson(), 2))
			print("\nComprobación: f(x) = 0\n", np.round(controlador.get_comprobacion_newton_raphson(), 2))

			# Mostrar resultados gráficos para la velocidad en x
			print("Gráfica de la velocidad en x:\n")
			controlador.graficar_velocidad_x_newton_raphson()

			# Mostrar resultados gráficos para la velocidad en y
			print("Gráfica de la velocidad en y:\n")
			controlador.graficar_velocidad_y_newton_raphson()
		

		elif opcion == "6":
			print("\n\n########################## Interpolación de resultados obtenidos con el método de Jacobi con sobrerelajación ##########################\n")

			factor_aumento = int(input("Ingrese el número de marcas que desea agregar entre cada par de puntos de la malla original:  "))
			print("")
			controlador.graficar_interpolacion_vx_jacobi(factor_aumento)
		

		elif opcion == "7":
			print("\n\n########################## Gráfico resumen interpolación de resultados obtenidos con el método de Jacobi con sobrerelajación ##########################\n")

			controlador.mostrar_resumen_interpolacion_vx_jacobi()
		

		elif opcion == "8":
			print("\n\n########################## Interpolación de resultados obtenidos con el método de Newton-Raphson ##########################\n")

			factor_aumento = int(input("Ingrese el número de marcas que desea agregar entre cada par de puntos de la malla original:  "))
			print("")
			controlador.graficar_interpolacion_vx_newton_raphson(factor_aumento)

		
		elif opcion == "9":
			print("\n\n########################## Gráfico resumen interpolación de resultados obtenidos con el método de Newton-Raphson ##########################\n")

			controlador.mostrar_resumen_interpolacion_vx_newton_raphson()


		elif opcion == "10":
			break

		else:
			print("\n\Opción inválida. Intente nuevamente.\n")


	print("\n\n########################## Fin del programa ##########################\n")
