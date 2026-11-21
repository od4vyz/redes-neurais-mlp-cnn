# Davi Lima de Oliveira - 15648741

import numpy as np
import os

# np.set_printoptions(threshold=1000, linewidth=75)
np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)

dataset = "CARACTERES"

# ======================================== FUNCOES MATEMATICAS ========================================
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

# ======================================== MULTILAYER PERCEPTRON ========================================
class MLP:
    def __init__(self, n_entrada, n_escondida, n_saida, alpha, max_epocas, erro_minimo, paciencia, seed=67):
        """Inicializa a arquitetura e os hiperparametros do MLP"""
        self.n_entrada = n_entrada # Numero de valores de entrada recebidos
        self.n_escondida = n_escondida # Numero de neuronios na camada escondida
        self.n_saida = n_saida # Numero de neuronios de saida
        self.alpha = alpha # Taxa de aprendizado
        self.max_epocas = max_epocas # Maximo de epocas a serem executadas
        self.erro_minimo = erro_minimo # Erro minimo aceitavel para interromper o treinamento
        self.paciencia = paciencia # Limite de epocas sem reducao de erro (parada antecipada)

        self.rng = np.random.default_rng(seed) # Gerador de valores aleatorios para inicializar os pesos

        # Inicializando as matrizes de peso e os bias
        self.v0 = self.rng.uniform(low=-0.5, high=0.5, size=(self.n_escondida,)) # Array de pesos do bias da camada escondida
        self.V = self.rng.uniform(low=-0.5, high=0.5, size=(self.n_entrada, self.n_escondida)) # Matriz de pesos V dos neuronios da camada escondida

        self.w0 = self.rng.uniform(low=-0.5, high=0.5, size=(self.n_saida,)) # Array de pesos do bias da camada de saida
        self.W = self.rng.uniform(low=-0.5, high=0.5, size=(self.n_escondida, self.n_saida)) # Matriz de pesos W dos neuronios da camada de saida

        # Salva os valores inicias para gerar os logs posteriormente
        self.v0_inicial = self.v0.copy()
        self.V_inicial = self.V.copy()
        self.w0_inicial = self.w0.copy()
        self.W_inicial = self.W.copy()

        self.erros_treino = []
        self.erros_validacao = []

    def feedforward(self, x):
        """Propaga a entrada e retorna as ativacoes da camada"""
        z_in = self.v0 + (x @ self.V)
        z = tanh(z_in)

        y_in = self.w0 + (z @ self.W)
        y = tanh(y_in)

        return z, y

    def backpropagation(self, x, z, y, t):

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
        self.w0 = self.w0 + correcao_w0
        self.W = self.W + correcao_w
        self.v0 = self.v0 + correcao_v0
        self.V = self.V + correcao_v

    def treinamento(self, X_treino, T_treino, X_validacao, T_validacao):
        epoca = 0 # inicializa a contagem de epocas
        mse_epoca = np.inf
        menor_erro = np.inf # Armazena o menor erro de validacao obtido ate o momento
        contador_paciencia = 0 # Contador de epocas sem melhoria na fase de validacao

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

# ======================================== FUNCOES AUXILIARES ========================================
def carregar_dados():
    """Carrega o dataset, embaralha as amostras e divide em treino, validacao e teste."""
    rng = np.random.default_rng(42)
    
    match dataset:
        case "CARACTERES":
            path_limpos = os.path.join(os.path.dirname(os.path.abspath(__file__)),"entradas", "caracteres-Fausett", "caracteres-limpo.csv")
            path_ruido = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entradas", "caracteres-Fausett", "caracteres-ruido.csv") 
            
            dados_limpos = np.loadtxt(path_limpos, delimiter=",", encoding="utf-8-sig")
            dados_ruido = np.loadtxt(path_ruido, delimiter=",", encoding="utf-8-sig")

            X_treino, T_treino = dados_limpos[:, :-7], dados_limpos[:, -7:] 
            X_validacao, T_validacao = dados_limpos[:, :-7], dados_limpos[:, -7:]  
            X_teste, T_teste = dados_ruido[:, :-7], dados_ruido[:, -7:] 

            return X_treino, T_treino, X_validacao, T_validacao, X_teste, T_teste

        case "CARACTERES_RUIDO":
            path_limpos = os.path.join(os.path.dirname(os.path.abspath(__file__)),"entradas", "caracteres-Fausett", "caracteres-limpo.csv")
            path_ruido = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entradas", "caracteres-Fausett", "caracteres_ruido20.csv") 
            
            dados_limpos = np.loadtxt(path_limpos, delimiter=",", encoding="utf-8-sig")
            dados_ruido = np.loadtxt(path_ruido, delimiter=",", encoding="utf-8-sig")

            X_treino, T_treino = dados_limpos[:, :-7], dados_limpos[:, -7:] 
            X_validacao, T_validacao = dados_limpos[:, :-7], dados_limpos[:, -7:]
            X_teste, T_teste = dados_ruido[:, :-7], dados_ruido[:, -7:] 

            return X_treino, T_treino, X_validacao, T_validacao, X_teste, T_teste

        case "COMPLETO":
            # Salva os caminhos dos arquivos a serem carregados
            path_entrada = os.path.join(os.path.dirname(os.path.abspath(__file__)),"entradas", "caracteres-completo", "X.npy")
            path_classes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entradas", "caracteres-completo", "Y_classe.npy") 

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

        case _:
            print("Dataset Invalido")

def escrever_txt(caminho_arquivo, conteudo):
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        f.write(conteudo)

def registrar_saida(rede, tipo_saida, dataset):
    path_saida = os.path.join(os.path.dirname(os.path.abspath(__file__)),"saidas", f"{dataset}")
    os.makedirs(path_saida, exist_ok=True)

    arquivos = {
        "hiperparametros": os.path.join(path_saida, "hiperparametros.txt"),
        "pesos_iniciais":  os.path.join(path_saida, "pesos_iniciais.txt"),
        "pesos_finais":    os.path.join(path_saida, "pesos_finais.txt"),
        "erro_treino":     os.path.join(path_saida, "erros_treino.txt"),
        "erro_validacao":  os.path.join(path_saida, "erros_validacao.txt"),
        "saidas_teste":    os.path.join(path_saida, "saidas_teste.txt"),
    }

    match tipo_saida:
        case "HIPERPARAMETRO":
            escrever_txt(arquivos["hiperparametros"], 
                         f"Neuronios de entrada: {rede.n_entrada}\n"
                         f"Neuronios escondidos: {rede.n_escondida}\n"
                         f"Neuronios de saida: {rede.n_saida}\n"
                         f"Taxa de Aprendizado (alpha): {rede.alpha}\n"
                         f"Máximo de Épocas: {rede.max_epocas}\n"
                         f"Erro mínimo aceitável: {rede.erro_minimo}\n"
                         f"Paciencia: {rede.paciencia}\n") 

        case "PESOS_INICIAIS":
            conteudo = f"Pesos iniciais da camada escondida (com Bias):\n{rede.v0_inicial}, {rede.V_inicial}\n\nPesos iniciais da camada de saída (com Bias):\n{rede.w0_inicial}, {rede.W_inicial}\n"
            escrever_txt(arquivos["pesos_iniciais"], conteudo) 

        case "PESOS_FINAIS":
            conteudo = f"Pesos finais da camada escondida (com Bias):\n{rede.v0}, {rede.V}\n\nPesos finais da camada de saída (com Bias):\n{rede.w0}, {rede.W}\n"
            escrever_txt(arquivos["pesos_finais"], conteudo) 
        
        case "ERRO":
            np.savetxt(arquivos["erro_treino"], rede.erros_treino, fmt="%.10f")
            np.savetxt(arquivos["erro_validacao"], rede.erros_validacao, fmt="%.10f")

        case "SAIDAS_TESTE":
            pass

        case _:
            pass

# ======================================== CODIGO PRINCIPAL ========================================
def main():
    X_treino, T_treino, X_validacao, T_validacao, X_teste, T_teste = carregar_dados()   

    rede = MLP(X_treino.shape[1], 20, T_treino.shape[1], 0.1, 500, 0.01, 10)

    registrar_saida(rede, "HIPERPARAMETRO", dataset)
    registrar_saida(rede, "PESOS_INICIAIS", dataset)

    rede.treinamento(X_treino, T_treino, X_validacao, T_validacao)

    registrar_saida(rede, "PESOS_FINAIS", dataset)
    registrar_saida(rede, "ERRO", dataset)

if __name__ == "__main__": main()