# Davi Lima de Oliveira - 15648741

import numpy as np
import os, time, string

# np.set_printoptions(threshold=1000, linewidth=75)
np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)

# ======================================== FUNCOES METRICAS ========================================
# ------------------------- Etapa de Treinamento -------------------------
def tanh(y_in):
    """Funcao de ativacao. Encapsulamento da Tangente Hiperbolica."""
    # A Tangente Hiperbolica foi escolhida como funcao de ativacao devido ao carater do problema, que ja esta modelado em escala bipolar [-1, 1]. A utilizacao de metodo ja existente no NumPy foi considerada necessaria devido a protecao contra overflow numerico existente para a funcao
    return np.tanh(y_in)

def derivada_tanh(y):
    """Calculo da derivada da funcao de ativacao."""
    # A ser utilizada no algoritmo de backpropagation. Recebe a saida ja ativada do neuronio
    return 1 - (y**2)

def mse(t, y):
    """Calculo de Erro Quadratico Medio"""
    # A ser utilizada no algoritmo de treinamento e validacao da rede. Recebe a saida esperada e a saida real dos neuronios
    return np.mean((t - y)**2)

# ------------------------- Etapa de Teste -------------------------
def calcular_acuracia(T_teste, Y):
    """Calcula a taxa de acerto global comparando as classes esperadas e previstas."""
    T_classes = np.argmax(T_teste, axis=1)
    Y_classes = np.argmax(Y, axis=1)
    
    acertos = T_classes == Y_classes
    acuracia = np.mean(acertos)

    return acuracia

def matriz_confusao(T_teste, Y):
    """Gera a matriz de confusao cruzando as classes esperadas com as previstas."""
    T_classes = np.argmax(T_teste, axis=1)
    Y_classes = np.argmax(Y, axis=1)

    n_classes = T_teste.shape[1]
    matriz = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(T_classes.shape[0]):
        matriz[T_classes[i]][Y_classes[i]] += 1

    return matriz

# ======================================== FUNCOES AUXILIARES ========================================
def carregar_dados():
    """Carrega o dataset, embaralha as amostras e divide em treino, validacao e teste."""
    rng = np.random.default_rng(42)
    
    # Salva os caminhos dos arquivos a serem carregados
    path_entrada = os.path.join(os.path.dirname(os.path.abspath(__file__)),"entradas", "X.npy")
    path_classes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entradas", "Y_classe.npy") 

    # Carrega todo o conjunto de dados em memoria
    dados_entrada_4d = np.load(path_entrada)
    dados_entrada = dados_entrada_4d.reshape(dados_entrada_4d.shape[0], -1) # remodela os dados de entrada para duas dimensões
    dados_classes = np.load(path_classes)

    # Divide o conjunto, mantendo apenas os dados a serem usados na etapa de treinamento
    entrada = dados_entrada[:-130, :]
    classes = dados_classes[:-130, :]

    # Redistribui os dados aleatoriamente para diminuir o fator de subrepresentacao de classes ao dividir o conjunto em treino e validacao
    permuta = rng.permutation(len(entrada))

    entrada = entrada[permuta]
    classes =  classes[permuta]

    # Delimita qual parte de cada conjunto sera utilizada em cada fase do algoritmo
    X_treino, T_treino = entrada[:-246, :], classes[:-246, :] # Aproximadamente 80% para treino
    X_validacao, T_validacao = entrada[-246:, :], classes[-246:, :] # Aproximadamente 20% para validação
    X_teste, T_teste = dados_entrada[-130:, :], dados_classes[-130:, :] # Os 130 ultimos padroes, contendo ruidos para teste

    return X_treino, T_treino, X_validacao, T_validacao, X_teste, T_teste

def caminho_saida(tipo_saida, rede):
    """Cria a pasta de saida e retorna o caminho completo do arquivo."""
    path_saida = os.path.join(os.path.dirname(os.path.abspath(__file__)),"saidas", f"z{rede.n_escondida}_a{str(rede.alpha).replace(".", "")}_p{rede.paciencia}_w{str(rede.high_pesos).replace(".", "")}")
    os.makedirs(path_saida, exist_ok=True)

    arquivo = os.path.join(path_saida, f"{tipo_saida}.txt")

    return arquivo

def escrever_txt(caminho_arquivo, conteudo):
    """Grava o texto formatado no arquivo de destino."""
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        f.write(conteudo)

def registrar_inicializacao(rede):
    """Salva os hiperparametros e os pesos iniciais da rede em arquivos de texto."""
    # Define os arquivos
    caminho_hiperparametros = caminho_saida("hiperparametros", rede)
    caminho_pesos_iniciais = caminho_saida("pesos_iniciais", rede)

    # Define o conteudo a ser registrado
    hiperparametros = f"========== Hiperparametros de Arquitetura e Inicialização ==========\nNeuronios de entrada: {rede.n_entrada}\nNeuronios escondidos: {rede.n_escondida}\nNeuronios de saida: {rede.n_saida}\nIntervalo de inicialização dos pesos: [{rede.low_pesos}, {rede.high_pesos}]\nSeed geradora: {rede.seed}\nTaxa de Aprendizado (alpha): {rede.alpha}\nMáximo de Épocas: {rede.max_epocas}\nErro mínimo aceitável: {rede.erro_minimo}\nPaciência: {rede.paciencia}"

    pesos_iniciais = f"========== Pesos iniciais da camada escondida (com Bias) ==========\n{rede.v0_inicial}, {rede.V_inicial}\n\n========== Pesos iniciais da camada de saída (com Bias) ==========\n{rede.w0_inicial}, {rede.W_inicial}\n"

    # Registra o conteudo
    escrever_txt(caminho_hiperparametros, hiperparametros) 
    escrever_txt(caminho_pesos_iniciais, pesos_iniciais) 

def registrar_treino(rede):
    """Exporta os pesos finais e os historicos de erro de treino e validacao."""
    # Define os arquivos    
    caminho_pesos_finais= caminho_saida("pesos_finais", rede)
    caminho_erro_treino = caminho_saida("erros_treino", rede)
    caminho_erro_validacao = caminho_saida("erros_validacao", rede)

    # Define o conteudo a ser registrado
    pesos_finais = f"========== Pesos finais da camada escondida (com Bias) ==========\n{rede.v0}, {rede.V}\n\n========== Pesos finais da camada de saída (com Bias) ==========\n{rede.w0}, {rede.W}\n"

    # Registra o conteudo
    escrever_txt(caminho_pesos_finais, pesos_finais) 
    np.savetxt(caminho_erro_treino, rede.erros_treino, fmt="%.10f")
    np.savetxt(caminho_erro_validacao, rede.erros_validacao, fmt="%.10f")

def registrar_metricas(rede, matriz, mse, acuracia, tempo, Y_teste, T_teste):
    """Registra as metricas de desempenho, a matriz de confusao e as saidas do teste."""
    # Define os arquivos    
    caminho_resultados = caminho_saida("resultados_teste", rede)
    caminho_saidas = caminho_saida("saidas_teste", rede)
    caminho_matriz = caminho_saida("matriz_confusao", rede)

    # Define o conteudo a ser registrado
    resultados = f"========== Resultados ==========\nErro Quadrático Médio: {mse}\nAcurácia: {acuracia} | {acuracia:.2%}\nÉpocas executadas: {len(rede.erros_treino)}\nMelhor Época: {rede.melhor_epoca}\nTempo de treinamento: {tempo:.6f} segundos"

    saidas_produzidas = []
    for i in range(Y_teste.shape[0]):
        conteudo = f"X {i}: | T: {np.argmax(T_teste[i])} ({string.ascii_uppercase[np.argmax(T_teste[i])]}) | Y: {np.argmax(Y_teste[i])} ({string.ascii_uppercase[np.argmax(Y_teste[i])]})| Saídas: {np.round(Y_teste[i], 4)}\n"
        saidas_produzidas.append(conteudo)

    # Registra o conteudo
    escrever_txt(caminho_resultados, resultados)  
    escrever_txt(caminho_saidas, "".join(saidas_produzidas))
    np.savetxt(caminho_matriz, matriz, fmt="%2d")

# ======================================== MULTILAYER PERCEPTRON ========================================
class MLP:
    def __init__(self, n_entrada, n_escondida, n_saida, alpha, max_epocas, erro_minimo, paciencia, seed=67, low_pesos=-0.5, high_pesos=0.5):
        """Inicializa a arquitetura e os hiperparametros do MLP"""
        self.n_entrada = n_entrada # Numero de valores de entrada recebidos
        self.n_escondida = n_escondida # Numero de neuronios na camada escondida
        self.n_saida = n_saida # Numero de neuronios de saida
        self.alpha = alpha # Taxa de aprendizado
        self.max_epocas = max_epocas # Maximo de epocas a serem executadas
        self.erro_minimo = erro_minimo # Erro minimo aceitavel para interromper o treinamento
        self.paciencia = paciencia # Limite de epocas sem reducao de erro (parada antecipada)

        self.seed = seed
        self.rng = np.random.default_rng(self.seed) # Gerador de valores aleatorios para inicializar os pesos

        # Inicializando as matrizes de peso e os bias
        self.low_pesos = low_pesos
        self.high_pesos = high_pesos
        self.v0 = self.rng.uniform(low=low_pesos, high=high_pesos, size=(self.n_escondida,)) # Array de pesos do bias da camada escondida
        self.V = self.rng.uniform(low=low_pesos, high=high_pesos, size=(self.n_entrada, self.n_escondida)) # Matriz de pesos V dos neuronios da camada escondida

        self.w0 = self.rng.uniform(low=low_pesos, high=high_pesos, size=(self.n_saida,)) # Array de pesos do bias da camada de saida
        self.W = self.rng.uniform(low=low_pesos, high=high_pesos, size=(self.n_escondida, self.n_saida)) # Matriz de pesos W dos neuronios da camada de saida

        # Salva os valores inicias para gerar os logs posteriormente
        self.v0_inicial = self.v0.copy()
        self.V_inicial = self.V.copy()
        self.w0_inicial = self.w0.copy()
        self.W_inicial = self.W.copy()

        self.erros_treino = []
        self.erros_validacao = []

        self.melhor_epoca = 0

    def feedforward(self, x):
        """Propaga a entrada e retorna as ativacoes da camada"""
        z_in = self.v0 + (x @ self.V)
        z = tanh(z_in)

        y_in = self.w0 + (z @ self.W)
        y = tanh(y_in)

        return z, y

    def backpropagation(self, x, z, y, t):
        """Calcula os gradientes de erro e ajusta os pesos e bias da rede."""
        # Calcula o erro da camada de saida e propaga para a camada escondida
        delta_k = (t - y) * derivada_tanh(y) # Calcula o erro da camada de saída multiplicando o erro bruto de cada neuronio pela sensibilidade do neuronio
        # utiliza esse erro para calcular a matriz de correcao dos pesos da camada
        correcao_w = self.alpha * np.outer(z, delta_k)
        correcao_w0 = self.alpha * delta_k

        # Calcula o erro da camada escondida e propaga para a camada de entrada
        delta_in_j = delta_k @ self.W.T # Calcula o erro bruto dos neuronios da camada escondida utilizando o erro da camada de saida e os pesos daquelas conexoes
        delta_j = delta_in_j * derivada_tanh(z) # multiplica o erro bruto pela sensibilidade para calcular o erro da camada escondida
        # utiliza esse erro para calcular a matriz de correcao dos pesos da camada
        correcao_v = self.alpha * np.outer(x, delta_j)
        correcao_v0 = self.alpha * delta_j

        # Recalcula e ajusta os pesos de toda a rede
        self.w0 += correcao_w0
        self.W += correcao_w
        self.v0 += correcao_v0
        self.V += correcao_v

    def treinamento(self, X_treino, T_treino, X_validacao, T_validacao):
        """Executa o loop de treinamento da rede com validacao e parada antecipada."""
        epoca = 0 # reinicializa a contagem de epocas
        mse_epoca = np.inf
        menor_erro = np.inf # Armazena o menor erro de validacao obtido ate o momento
        contador_paciencia = 0 # Contador de epocas sem melhoria na fase de validacao

        self.melhor_epoca = 0
        melhor_v0 = self.v0.copy()
        melhor_V = self.V.copy()
        melhor_w0 = self.w0.copy()
        melhor_W = self.W.copy()

        self.erros_treino = []
        self.erros_validacao = []

        while epoca < self.max_epocas and mse_epoca > self.erro_minimo and contador_paciencia < self.paciencia:
            epoca += 1 
            erros_treino = 0

            # Randomiza a ordem dos padroes a cada nova epoca para evitar que a rede vicie na sequencia dos dados
            permuta = self.rng.permutation(len(X_treino)) # Gera a ordem aleatoria de indices
            X_treino = X_treino[permuta]
            T_treino = T_treino[permuta]

            for i in range(X_treino.shape[0]):
                z_treino, y_treino = self.feedforward(X_treino[i])
                self.backpropagation(X_treino[i], z_treino, y_treino, T_treino[i])
                erros_treino += mse(T_treino[i], y_treino)

            mse_epoca = erros_treino / X_treino.shape[0]
            self.erros_treino.append(mse_epoca)

            # Fase de validacao da rede                        
            z, y = self.feedforward(X_validacao)
            mse_validacao = mse(T_validacao, y)
            self.erros_validacao.append(mse_validacao)
            if mse_validacao < menor_erro:
                menor_erro = mse_validacao
                contador_paciencia = 0
                self.melhor_epoca = epoca

                melhor_v0 = self.v0.copy()
                melhor_V = self.V.copy()
                melhor_w0 = self.w0.copy()
                melhor_W = self.W.copy()
            else:
                contador_paciencia += 1

        # Restaura os melhores pesos encontrados para serem os pesos finais da rede
        self.v0 = melhor_v0
        self.V = melhor_V
        self.w0 = melhor_w0
        self.W = melhor_W
        
    def executar(self, X):
        """Aplica as entradas na rede treinada e retorna as saidas produzidas."""
        y = self.feedforward(X)[1]

        return y
    
# ======================================== CODIGO PRINCIPAL ========================================
def main():
    """Organiza o carregamento dos dados, treinamento, avaliacao e registro de saidas."""
    # Definindo os hiperparametros da rede
    n_escondida = 60
    alpha = 0.01
    max_epocas = 5000
    erro_minimo = 0.0
    paciencia = 50

    inicio_execucao = time.perf_counter()
    # Carrega o dataset em memoria e particiona
    X_treino, T_treino, X_validacao, T_validacao, X_teste, T_teste = carregar_dados()   
    print("Dataset carregado.")

    # Etapa de inicializacao da rede e registro dos parametros iniciais
    print("Inicializando rede.")
    rede = MLP(X_treino.shape[1], n_escondida, T_treino.shape[1], alpha, max_epocas, erro_minimo, paciencia)

    print("Registrando hiperparâmetros e pesos iniciais.")
    registrar_inicializacao(rede)

    # Etapa de treinamento da rede
    print("Treinamento iniciado.")
    inicio_treinamento = time.perf_counter()
    rede.treinamento(X_treino, T_treino, X_validacao, T_validacao)
    tempo_treinamento = time.perf_counter()-inicio_treinamento
    print(f"Treinamento finalizado. Tempo total: {tempo_treinamento}")

    print("Registrando Pesos e Erro")
    registrar_treino(rede)

    # Etapa de teste da rede, registrando os resultados
    print("Executando testes")
    Y_teste = rede.executar(X_teste) # Executa o teste da rede com os dados de teste 

    matriz = matriz_confusao(T_teste, Y_teste) # Gera a matriz de confusao
    mse_teste = mse(T_teste, Y_teste) # Calcula o Erro Quadratico Medio da rede
    acuracia = calcular_acuracia(T_teste, Y_teste) # Calcula a acuracia da rede

    print("Registrando resultados")
    registrar_metricas(rede, matriz, mse_teste, acuracia, tempo_treinamento, Y_teste, T_teste)

    print(f"Execução finalizada. Tempo total: {time.perf_counter()-inicio_execucao}")

if __name__ == "__main__": main()