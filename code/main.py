from red_neuronal import *

def main():

	# Se inicia la red neuronal.
	red_neuronal = Red([2, 3, 1])

	# Valores de entrada de aprendizaje.
	entradas = np.array(
				[[0, 0], 
				[0, 0.687],  
				[0, -1], 
				[0.7624, 0],
				[0.5, 1], 
				[0.5, -1], 
				[1, 1],    
				[1, -1],
				[0.7, 0.2],
				[0.1, 0.2],
				[0.2, 0.7],
				[-0.6, 0.24],
				[0.456, -0.784],
				[0.45, 1],
				[0.34172, 0.6617],
				[0.6, 0.7],
				[0.47, 1],
				[-1, 0.76],
				[0.5, -1],
				[-0.789, 0.5676],
				[0.2, -1],
				[-0.5679, 1],
				[0.156, 0.1687],
				[0.5728, 0.2591],
				[0.6416, 0.1179]])

	# Valores de salida para las entradas dadas arriba.
	salidas = np.array([
				0,
				0,   
				0,
				0,
				0.5,  
				-0.5,  
				1,  
				-1,
				0.14,
				0.02,
				0.14, 
				-0.144,
				-0.357,
				0.45,
				0.226116,
				0.42, 
				0.47,
				-0.76,
				-0.5,
				-0.4478364,
				-0.2, 
				-0.5679,
				0.0263172,
				0.148412, 
				0.075644]) 

	# En total son 25 entradas de entrenamiento.
	# Aquí se le mandan los parámetros a la red para que aprenda.
	red_neuronal.entrenar(entradas, salidas)
	
	# Una vez aprendidos los valores, se muestran los mismos 25 valores pero llamando al método estimador.
	# Esto para ver la presición de la red neuronal con datos que ya conoce.
	print("\n\nVALORES ESTIMADOS PARA DATOS CONOCIDOS:")
	i = 0
	for entrada in entradas:
		estimado = red_neuronal.estimar(entrada)
		print("Valor de x: " + str(entrada[0]) + ", valor de y: " + str(entrada[1]) + "\t| valor real = " + str (salidas[i]) + " - estimado: " + str(estimado[0]))
		i += 1

	# Se crean unos nuevos parámetros de entrada que la red nunca ha visto para predecir el resultado.
	nuevos = np.array([
		[0.4, 1],
		[0.3, -1],
		[0.567, 0.2517],
		[1, 0.78],
		[0.121, 0.231],
		[0.1212, -1],
		[0.6611, 0.13619]])

	diferenciaTotal = 0
	numeroValores = 0
	
	print("\n\nVALORES ESTIMADOS PARA DATOS DESCONOCIDOS:")
	for nuevo in nuevos:
		estimado = red_neuronal.estimar(nuevo)
		print("Valor de x: " + str(nuevo[0]) + ", valor de y: " + str(nuevo[1]) + "\t| valor real = " + str (nuevo[0] * nuevo[1]) + " - estimado: " + str(estimado[0]))

		numeroValores += 1
		diferenciaTotal += (nuevo[0] * nuevo[1]) - estimado[0]

	print("\n")
	print("Promedio de diferencia: " + str (diferenciaTotal / numeroValores))

main()
