import math
import random
import numpy as np

def carregar_dados(caminho_arquivo):
    with open(caminho_arquivo, 'r') as file:
        linhas = file.readlines()
    dados = []
    for linha in linhas:
        valores = linha.strip().split(',')
        dados.append(list(map(float, valores[3:6])))  # Usando apenas os atributos 4, 5, 6 (índices 3, 4, 5)
    return dados

def carregar_dados_saida(caminho_arquivo):
    with open(caminho_arquivo, 'r') as file:
        linhas = file.readlines()
    dados = []
    for linha in linhas:
        valores = linha.strip().split(',')
        array = [0, 0, 0, 0]  # Cria uma lista com 4 posições, todas inicializadas com 0
        array[int(valores[7]) - 1] = 1  # Define o valor 1 na posição não zero
        dados.append(array)  # Usando apenas os atributos 4, 5, 6 (índices 3, 4, 5)
    return dados
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_neurons, hidden_neurons, output_neurons):
    input_hidden_weights = [[random.uniform(-1, 1) for _ in range(hidden_neurons)] for _ in range(input_neurons)]
    hidden_output_weights = [[random.uniform(-1, 1) for _ in range(output_neurons)] for _ in range(hidden_neurons)]
    return input_hidden_weights, hidden_output_weights

def train_neural_network(inputs, targets, input_hidden_weights, hidden_output_weights, learning_rate, epochs):
    for epoch in range(epochs):
        total_error = 0
        for i, input_data in enumerate(inputs):
            # Feedforward
            hidden_inputs = [sum([input_data[j] * input_hidden_weights[j][k] for j in range(len(input_data))]) for k in range(len(input_hidden_weights[0]))]
            hidden_outputs = [sigmoid(val) for val in hidden_inputs]

            final_inputs = [sum([hidden_outputs[j] * hidden_output_weights[j][k] for j in range(len(hidden_outputs))]) for k in range(len(hidden_output_weights[0]))]
            final_outputs = [sigmoid(val) for val in final_inputs]

            # Calcula o erro
            error = [targets[i][j] - final_outputs[j] for j in range(len(targets[i]))]
            total_error += sum([e ** 2 for e in error])

            # Backpropagation
            output_deltas = [error[j] * sigmoid_derivative(final_outputs[j]) for j in range(len(error))]
            hidden_deltas = [sum([output_deltas[k] * hidden_output_weights[j][k] for k in range(len(output_deltas))]) * sigmoid_derivative(hidden_outputs[j]) for j in range(len(hidden_outputs))]

            # Atualiza os pesos (somente a cada tantas iterações)
            if (i + 1) % update_interval == 0:
                for j in range(len(hidden_output_weights)):
                    for k in range(len(hidden_output_weights[0])):
                        hidden_output_weights[j][k] += learning_rate * output_deltas[k] * hidden_outputs[j]
                for j in range(len(input_hidden_weights)):
                    for k in range(len(input_hidden_weights[0])):
                        input_hidden_weights[j][k] += learning_rate * hidden_deltas[k] * input_data[j]

        print(f'Época {epoch + 1}/{epochs}')

    return input_hidden_weights, hidden_output_weights

def predict(input_data, input_hidden_weights, hidden_output_weights):
    hidden_inputs = [sum([input_data[j] * input_hidden_weights[j][k] for j in range(len(input_data))]) for k in range(len(input_hidden_weights[0]))]
    hidden_outputs = [sigmoid(val) for val in hidden_inputs]

    final_inputs = [sum([hidden_outputs[j] * hidden_output_weights[j][k] for j in range(len(hidden_outputs))]) for k in range(len(hidden_output_weights[0]))]
    final_outputs = [sigmoid(val) for val in final_inputs]

    # Converte as saídas para as classes (1 a 4)
    predicted_class = final_outputs.index(max(final_outputs)) + 1
    return predicted_class


# Dados de treino e saída esperada
inputs = np.array(carregar_dados('treino_sinais_vitais_com_label.txt'))
targets = np.array(carregar_dados_saida('treino_sinais_vitais_com_label.txt'))

# Parâmetros da rede neural
input_neurons = 3
hidden_neurons = 7  # Número de neurônios na camada oculta
output_neurons = 4
learning_rate = 0.01
epochs = 20000
update_interval = 10  # Intervalo para atualização dos pesos

# Inicialização dos pesos
input_hidden_weights, hidden_output_weights = initialize_weights(input_neurons, hidden_neurons, output_neurons)

# Treinamento da rede neural
input_hidden_weights, hidden_output_weights = train_neural_network(inputs, targets, input_hidden_weights, hidden_output_weights, learning_rate, epochs)

new_data = np.array(carregar_dados('treino_sinais_vitais_com_label.txt'))
for i,data in new_data:
    predicted_class = predict(data, input_hidden_weights, hidden_output_weights)
    print(f'Linha: {i} e Classe prevista: {predicted_class}')