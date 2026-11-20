# Davi Lima de Oliveira - 15648741

import numpy as np
import os

# --------------- FUNCOES MATEMATICAS ---------------




# --------------- MULTILAYER PERCEPTRON ---------------
class MLP:
    def __init__(self, ):
        pass
# --------------- FUNCOES AUXILIARES ---------------
dataset = "XOR"

match dataset:
    case "XOR":
        caminho_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entradas/portas-logicas", "problemXOR.csv") 
        dados = np.loadtxt(caminho_csv, delimiter=",", encoding="utf-8-sig")

        X_treino = dados[:, :-1]
        T_treino = dados[:, -1:]
        X_validacao = None
        T_validacao = None 
        X_teste = dados[:, :-1]
        T_teste = dados[:, -1:]

        return X_treino, T_treino, X_validacao, T_validacao, X_teste, T_teste    
    case "CARACTERES":
        pass
    case "COMPLETO":
        pass
    case _:
        print("Dataset Invalido")

# --------------- CODIGO PRINCIPAL ---------------
def main():
    pass   

if __name__ == "__main__": main()