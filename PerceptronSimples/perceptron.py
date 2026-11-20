import numpy as np
import os

def funcao_bipolar(y_in):
    if y_in >= 0:
        return 1
    else:
        return -1

class Perceptron:
    def __init__(self, n_entradas, alpha, max_epocas):
        self.b = 0
        self.w = np.zeros(n_entradas)
        self.alpha = alpha 
        self.max_epocas = max_epocas   

    def feedforward(self, x):
        y_in = self.b + np.dot(self.w, x)
        y = funcao_bipolar(y_in)
        return y

    def treinamento(self, X, T):
        erro = True
        epoca = 0
        historico_erros = []
        while erro and epoca != self.max_epocas:
            erro = False
            contagem_erros = 0
            epoca += 1
            for i in range(X.shape[0]):
                x = X[i]
                t = T[i]
                y = self.feedforward(x)
                if y != t:
                    contagem_erros += 1
                    self.w = self.w + self.alpha * t * x
                    self.b = self.b + self.alpha * t
                    erro = True
                historico_erros.append(contagem_erros)
        return historico_erros

# ------------------Código Principal------------------

porta_atual = "XOR"

# Leitura da planilha e importação dos dados
caminho_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entradas", "problemXOR.csv") # Seleciona o arquivo de entrada

dados = np.loadtxt(caminho_csv, delimiter=",", encoding="utf-8-sig")

X = dados[:, :-1] # fatia a matriz de dados salvando apenas as entradas 
T = dados[:, -1] # fatia a matriz de dados salvando apenas as saidas
n_entradas = X.shape[1]

# Preparando arquivos de saída
pasta_saida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saidas")
os.makedirs(pasta_saida, exist_ok=True)

arquivos = {
    "hiperparametros": os.path.join(pasta_saida, "hiperparametros.txt"),
    "pesos_iniciais":  os.path.join(pasta_saida, "pesos_iniciais.txt"),
    "pesos_finais":    os.path.join(pasta_saida, "pesos_finais.txt"),
    "erros":           os.path.join(pasta_saida, "erros_treino.txt"),
    "saidas_teste":    os.path.join(pasta_saida, "saidas_teste.txt"),
}

def salvar_arquivo(caminho_arquivo: str, conteudo: str):
    with open(caminho_arquivo, "a", encoding="utf-8") as f:
        f.write(conteudo)

# Instanciando o Neurônio
neuronio = Perceptron(n_entradas=n_entradas, alpha=0.1, max_epocas=500)

salvar_arquivo(arquivos["hiperparametros"], f"{porta_atual}\nalpha: {neuronio.alpha}\nmax_epocas: {neuronio.max_epocas}\n\n") # registrando os hiperparametros
salvar_arquivo(arquivos["pesos_iniciais"], f"{porta_atual}\n{neuronio.b}\n{neuronio.w}\n\n") # registra os pesos iniciais

# Registrando o treinamento
historico_erros = neuronio.treinamento(X, T)
salvar_arquivo(arquivos["erros"], f"{porta_atual}\n{historico_erros}\n\n") # registra o historico de erros

salvar_arquivo(arquivos["pesos_finais"], f"{porta_atual}\n{neuronio.b}\n{neuronio.w}\n\n") # registra os pesos finais

# Testando o Neurônio
salvar_arquivo(arquivos["saidas_teste"], f"{porta_atual}\n\n")
for i in range(X.shape[0]):
    x = X[i]
    t = T[i]
    y = neuronio.feedforward(x)
    salvar_arquivo(arquivos["saidas_teste"], f"{x}, {t}, {y}\n\n")






