from math import log2

class Nó:
    def __init__(self, atributo=None, resultado=None):
        self.atributo = atributo  # Atributo usado para dividir
        self.resultado = resultado  # Classe atribuída ao nó folha
        self.filhos = {}  # Dicionário para armazenar os filhos do nó

    def __str__(self, nivel=0):
        resultado = str(self.resultado) if self.resultado is not None else ""
        representação = "\t" * nivel + f"Atributo: {self.atributo}, Resultado: {resultado}\n"
        for valor, filho in self.filhos.items():
            representação += "\t" * (nivel + 1) + f"Valor: {valor} --> "
            representação += filho.__str__(nivel + 2)
        return representação

# Função para calcular a entropia de uma coluna
def calcular_entropia_coluna(coluna):
    contagens = {}
    total = len(coluna)
    for valor in coluna:
        contagens[valor] = contagens.get(valor, 0) + 1

    entropia = 0
    for count in contagens.values():
        probabilidade = count / total
        entropia -= probabilidade * log2(probabilidade)

    return entropia

# Função para calcular o ganho de informação de um atributo
def calcular_ganho_informacao(data, coluna_alvo):
    entropia_total = calcular_entropia_coluna(coluna_alvo)
    entropia_atributo = 0
    total = len(coluna_alvo)
    valores_unicos = set(coluna_alvo)

    for valor in valores_unicos:
        indices_subset = [i for i, v in enumerate(coluna_alvo) if v == valor]
        entropia_subset = calcular_entropia_coluna([data[i] for i in indices_subset])
        peso_subset = len(indices_subset) / total
        entropia_atributo += peso_subset * entropia_subset

    ganho_informacao = entropia_total - entropia_atributo
    return ganho_informacao


# Função para construir a árvore de decisão usando ID3
def construir_arvore_decisao(data, atributos, coluna_alvo):
    if len(set(coluna_alvo)) == 1:
        return Nó(resultado=coluna_alvo[0])

    if len(atributos) == 0:
        classe_majoritaria = max(set(coluna_alvo), key=coluna_alvo.count)
        return Nó(resultado=classe_majoritaria)

    melhor_atributo_idx = max(range(len(atributos)),
                              key=lambda i: calcular_ganho_informacao([row[i] for row in data], coluna_alvo))

    nó_atual = Nó(atributo=melhor_atributo_idx)
    valores_unicos = set([row[melhor_atributo_idx] for row in data])
    for valor in valores_unicos:
        indices_subset = [i for i, row in enumerate(data) if row[melhor_atributo_idx] == valor]
        subset_data = [data[i] for i in indices_subset]
        subset_atributos = [attr for i, attr in enumerate(atributos) if i != melhor_atributo_idx]
        subset_coluna_alvo = [coluna_alvo[i] for i in indices_subset]
        nó_atual.filhos[valor] = construir_arvore_decisao(subset_data, subset_atributos, subset_coluna_alvo)

    return nó_atual


# Função para prever a classe usando a árvore de decisão
def prever_classe(nó, amostra):
    if nó.resultado is not None:
        return nó.resultado
    valor_atributo = amostra[nó.atributo]
    filho = nó.filhos.get(valor_atributo)
    if filho is None:
        return max(nó.filhos.values(), key=lambda x: x.resultado).resultado
    return prever_classe(filho, amostra)


# Carregar dados de um arquivo de texto
def carregar_dados(caminho_arquivo):
    with open(caminho_arquivo, 'r') as file:
        linhas = file.readlines()
    dados = []
    for linha in linhas:
        valores = linha.strip().split(',')
        dados.append(list(map(float, valores[3:6])))  # Usando apenas os atributos 4, 5, 6 (índices 3, 4, 5)
    return dados


# Leitura dos dados de treino com rótulos
dados_com_rotulos = carregar_dados('treino_sinais_vitais_com_label.txt')
coluna_alvo = [int(linha.strip().split(',')[-1]) for linha in open('treino_sinais_vitais_com_label.txt', 'r')]

# Leitura dos dados de teste sem rótulos
dados_teste = carregar_dados('treino_sinais_vitais_sem_label.txt')

# Atributos da árvore (índices 0, 1, 2 correspondem a qPa, pulso, frequencia_respiratoria)
atributos = [0, 1, 2]

# Construção da árvore de decisão
arvore_decisao = construir_arvore_decisao(dados_com_rotulos, atributos, coluna_alvo)

print(arvore_decisao)

# Previsão da classe usando a árvore de decisão
classes_previstas = [prever_classe(arvore_decisao, amostra) for amostra in dados_teste]

# Resultados da previsão
for amostra, classe_prevista in zip(dados_teste, classes_previstas):
    print(f"Linha: {', '.join(map(str, amostra))}, Classe Prevista: {classe_prevista}")
