import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Funções de Ativação

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dicionários de funções de ativação
activation_functions = {
    'sigmoid': sigmoid
}

# Funções de Perda
def binary_loss(y, output):
    epsilon = 1e-15
    output = np.clip(output, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))

# Dicionário de funções de perda
loss_functions = {
    'binary': binary_loss
}

# Classe NeuralNetwork
class NeuralNetwork:
    def __init__(self, input_size, hidden_neurons, output_neurons, activation, loss, learning_rate, task_type):
        self.input_size = input_size
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.activation = activation_functions[activation]
        self.loss = loss_functions[loss]
        self.learning_rate = learning_rate
        self.task_type = task_type

        # Inicialização dos pesos
        self.W1 = np.random.randn(input_size, hidden_neurons) * np.sqrt(2 / input_size)
        self.B1 = np.zeros((1, hidden_neurons))
        self.W2 = np.random.randn(hidden_neurons, output_neurons) * np.sqrt(2 / hidden_neurons)
        self.B2 = np.zeros((1, output_neurons))
    
# Função para carregar os dados
def load_data(file_path, task_type):
    data = pd.read_csv(file_path)
    if task_type == 'binary':
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        y = y.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test, None, None
    
    elif task_type == 'regression':
        dados_multi = data[['TV', 'Radio', 'Newspaper', 'Sales']]
        # Separação das variáveis independentes (X) e dependente (y)
        X_multi = dados_multi[['TV', 'Radio', 'Newspaper']].values
        y_multi = dados_multi['Sales'].values
        # Normaliza as variáveis de entrada (X)
        scaler_X = StandardScaler()
        X_multi_scaled = scaler_X.fit_transform(X_multi)
        # Normaliza a variável de saída (y)
        scaler_y = StandardScaler()
        y_multi_scaled = scaler_y.fit_transform(y_multi.reshape(-1, 1)).flatten()

        # Divide os dados em treino e teste (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_multi_scaled, y_multi_scaled, test_size=0.2, random_state=0
        )
        return X_train, X_test, y_train, y_test, scaler_y.mean_[0], scaler_y.scale_[0]

    elif task_type == 'multiclass':
        target = data.pop('Species')  # Ajustar para a coluna correta do dataset Iris
        class_map = {label: idx for idx, label in enumerate(target.unique())}
        target_encoded = target.map(class_map).values
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(data)

        # Dividir em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            dados_normalizados,
            target_encoded,
            test_size=0.2,
            random_state=0,
            stratify=target_encoded
        )
        return X_train, X_test, y_train, y_test, None, None  # Saída direta como inteiros

if __name__ == "__main__":
    print("Selecione o tipo de tarefa:")
    print("1: Classificação Binária (heart binary.csv)")
    print("2: Regressão (advertisement.csv)")
    print("3: Classificação Multiclasse (iris.csv)")

    try:
        choice = int(input("Digite o número correspondente à sua escolha: "))
    except ValueError:
        print("Entrada inválida. Por favor, digite um número válido.")
        exit()

    if choice == 1:
        dataset_path = "heart binary.csv"
        task_type = 'binary'
        activation = 'sigmoid'
        loss = 'binary'
        output_neurons = 1
        hidden_neurons = 200
        learning_rate = 0.005
        print(f"Carregando dados para a tarefa: {task_type}...")
        X_train, X_test, y_train, y_test, y_mean, y_std = load_data(dataset_path, task_type)

    else:
        print("Escolha inválida.")
        exit()

    print("Configurando o modelo...")
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_neurons=hidden_neurons,
        output_neurons=output_neurons,
        activation=activation,
        loss=loss,
        learning_rate=learning_rate,
        task_type=task_type
    )

    print("Iniciando o treinamento...")
    model.train(X_train, y_train, X_test, y_test, epochs=1500)