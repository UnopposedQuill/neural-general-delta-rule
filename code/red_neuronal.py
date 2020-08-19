from funciones import *

class Red:
 
    def __init__(self, capas, activation='tanh'):

        self.capas = capas
        self.pesos = []
        self.deltas = []
        self.llenar_pesos()

    """
        Método que llena la red neuronal.
        La red es de tipo [2, 3, 1].
        Es decir, 2 nodos de entrada, 3 en capa oculta y una salida solamente.
        Los pesos_rnd son valores que se generan tomando en cuenta las relaciones de las matrices 2x3 y 3x1.
    """
    def llenar_pesos(self):
        for i in range(1, len(self.capas) - 1):
            self.pesos.append(2 * np.random.random((self.capas[i], self.capas[i] + 1)) -1)
        self.pesos.append(2 * np.random.random((self.capas[i] + 1, self.capas[i + 1])) - 1)

    """
        Método que entrena la red neuronal.
    """
    def entrenar(self, entradas, salidas):
        
        # Se inicializa la variable inicial con la función ones de Numpy que llena el dato con unos.
        # Inicialización de Bias (en las entradas).
        inicial = np.atleast_2d(np.ones(entradas.shape[0]))
        entradas = np.concatenate((inicial.T, entradas), 1)

        # Inicialización de valores importantes:
        # Tasa de aprendizaje: sirve para "amortiguar" el cambio de los valores de los pesos.
        # Iteraciones: número de veces que la red neuronal itera sobre el conjunto inicial de datos para aprender.
        tasa_aprendizaje = 0.05, 
        iteraciones = 15000
        
        # Iteraciones de aprendizaje de la red.
        for iteracion in range(iteraciones):
            i = np.random.randint(entradas.shape[0])
            actual = [entradas[i]]
 
            # Iteración de la capa de entrada.
            # Pesos de enlace de entrada y de oculta.
            for index_peso in range(len(self.pesos)):
                    dot_value = np.dot(actual[index_peso], self.pesos[index_peso])
                    activation = tangente(dot_value)
                    actual.append(activation)

            # En esta parte se calcula el error.
            # A este error se le aplica un cálculo que involucra la derivada de la función de aprendizaje.
            error = salidas[i] - actual[-1]
            deltas = [error * derivada_tangente(actual[-1])]

            # Iteración de la capa oculta.
            # Pesos de enlace de oculta y salida.
            for l in range(len(actual) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.pesos[l].T) * derivada_tangente(actual[l]))
            self.deltas.append(deltas)

            # Inversión para el backpropagation.
            deltas.reverse()
 
            # Aplicación del Backpropagation.
            # Se realiza lo siguiente:
            #   * Se multiplican los delta resultantes con las activaciones entrantes, esto para obtener la gradiente de peso.
            #   * Se actualiza el peso disminuyendolo por el porcentaje de gradiente.
            for i in range(len(self.pesos)):
                capa = np.atleast_2d(actual[i])
                delta = np.atleast_2d(deltas[i])
                self.pesos[i] += tasa_aprendizaje * capa.T.dot(delta)
 
            if iteracion % 500 == 0: print('Iteraciones: ', iteracion)
 
    """
        Método que se encarga de predecir un resultado dada una entrada.
        Este método es útil luego de que la red neuronal haya sido entrenada.
        Aquí lo que se hace es recuperar el valor de entrada formato [x, y] y aplicar el algoritmo de activación
        con los pesos que ya se aprendieron en el entrenamiento.
        Se retorna un único valor de salida y es el estimado de f(x, y) = xy.
    """
    def estimar(self, x):
        salida = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for i in range(0, len(self.pesos)):
            salida = tangente(np.dot(salida, self.pesos[i]))
        return salida
