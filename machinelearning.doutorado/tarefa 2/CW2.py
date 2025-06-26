# %% [markdown]
# # Sistema de Recomendação - MovieLens 100K
#
# Este notebook implementa três sistemas de recomendação diferentes para recomendar filmes aos usuários.
# Usando o dataset MovieLens 100K contido no arquivo "U.csv".

# %%
# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Carregamento e Análise Inicial dos Dados

# %%
# Carregamento dos dados como numpy array - arquivo contém apenas dados sem headers
user_movie_matrix = np.loadtxt('U.csv', delimiter=',')
print("\nPrimeiras linhas e colunas:")
print(user_movie_matrix[:5, :5])

# %%
# Verificação da esparsidade dos dados
total_entries = user_movie_matrix.shape[0] * user_movie_matrix.shape[1]
non_zero_entries = np.count_nonzero(user_movie_matrix)
sparsity = 1 - (non_zero_entries / total_entries)

print(f"\nEstatísticas da matriz:")
print(f"Número de usuários: {user_movie_matrix.shape[0]}")
print(f"Número de filmes: {user_movie_matrix.shape[1]}")
print(f"Entradas não-zero: {non_zero_entries}")
print(f"Esparsidade: {sparsity:.4f} ({sparsity*100:.2f}%)")

# %% [markdown]
# ### 1. Funções para Análise de Distribuições

# %%
def plot_ratings_histograms(matrix, title_suffix=""):
    """
    Plota histogramas de ratings por usuário e por filme lado a lado
    """
    # Contagem de ratings por usuário
    ratings_per_user = np.count_nonzero(matrix, axis=1)

    # Contagem de ratings por filme
    ratings_per_movie = np.count_nonzero(matrix, axis=0)

    # Criando subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histograma de ratings por usuário
    ax1.hist(ratings_per_user, bins=60, color='green')
    ax1.set_xlabel('Contagem de Ratings dados')
    ax1.set_ylabel('Número de Usuários')
    ax1.set_title(f'Distribuição de Ratings por Usuário{title_suffix}')
    ax1.grid(True, alpha=0.3)

    # Histograma de ratings por filme
    ax2.hist(ratings_per_movie, bins=60, color='orange')
    ax2.set_xlabel('Contagem de Ratings recebidos')
    ax2.set_ylabel('Número de Filmes')
    ax2.set_title(f'Distribuição de Ratings por Filme{title_suffix}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
# %%
# Plotando histogramas para o dataset original
print("Análise do Dataset Original:")
plot_ratings_histograms(user_movie_matrix, " (Dataset Original)")

# %% [markdown]
# ### 2. Divisão dos Dados em Treino, Validação e Teste

# %%
# Move para o conjunto certo, retirando do treino, verificando se a coluna não ficará vazia
def move_ratings(user_idx, train_matrix, to_matrix, to_move_idx, remaining_movies, min_rated_per_movie = 1):
    """
    Move ratings de um usuário da matriz de treino para outra matriz (validação ou teste),
    garantindo que nenhuma coluna fique sem ratings.
    """
    for idx in to_move_idx:
        # Verifica se há mais de 1 rating para este filme
        if np.count_nonzero(train_matrix[:, idx]) > min_rated_per_movie:
            to_matrix[user_idx, idx] = train_matrix[user_idx, idx]
            train_matrix[user_idx, idx] = 0
        else:
            if len(remaining_movies) > 0:
                print(f"Aviso: Filme {idx} não pode ser movido (ficaria sem ratings). Escolhendo substituto...")
                new_movie_idx = np.random.choice(remaining_movies)
                remaining_movies.remove(new_movie_idx)
                train_matrix, to_matrix, remaining_movies = move_ratings(user_idx, train_matrix, to_matrix, [new_movie_idx], remaining_movies, min_rated_per_movie)
            else:
                print(f"Aviso: Não há filmes restantes para substituir o filme {idx}")

    return train_matrix, to_matrix, remaining_movies

def train_test_split(matrix, n_val=5, n_test=5):
    """
    Divide os dados em conjuntos de treino, validação e teste.
    Para cada usuário, remove aleatoriamente 10 ratings: 5 para validação e 5 para teste.
    """
    train_matrix = matrix.copy()
    val_matrix = np.zeros_like(train_matrix)
    test_matrix = np.zeros_like(train_matrix)

    # Para cada usuário
    for user_idx in range(matrix.shape[0]):
        rated_movies = np.where(matrix[user_idx] > 0)[0]
        selected_movies = np.random.choice(rated_movies, size=n_val+n_test, replace=False)
        remaining_movies = [m for m in rated_movies if m not in selected_movies]

        # Dividir em validação (primeiros 5) e teste (últimos 5)
        val_movies = selected_movies[:n_val]
        test_movies = selected_movies[n_val:]

        train_matrix, val_matrix, remaining_movies = move_ratings(user_idx, train_matrix, val_matrix, val_movies, remaining_movies)
        train_matrix, test_matrix, remaining_movies = move_ratings(user_idx, train_matrix, test_matrix, test_movies, remaining_movies)

    # Verificar estatísticas finais
    ratings_per_movie = np.count_nonzero(train_matrix, axis=0)
    print(f"Filme com menos ratings tem {np.min(ratings_per_movie)} ratings")

    return train_matrix, val_matrix, test_matrix

# %%
# Configurar seed para reprodutibilidade
np.random.seed(42)

# Dividir os dados
train_matrix, val_matrix, test_matrix = train_test_split(user_movie_matrix)

# %% [markdown]
# ## 2. Implementação do Sistema de Recomendação usando NMF

# %%
def nmf(matrix, k, max_iter=1000, tol=0.01, alpha=0.0001, beta=0.0001, epsilon=1e-9):
    """
    Implementa o algoritmo Non-negative Matrix Factorization (NMF) com regularização.

    Parâmetros:
    - matrix: matriz de treino Y (usuários x filmes)
    - k: número de fatores latentes
    - max_iter: número máximo de iterações
    - tol: tolerância para convergência
    - alpha: parâmetro de regularização para W
    - beta: parâmetro de regularização para H
    - epsilon: pequeno valor para evitar divisão por zero

    Retorna:
    - W: matriz de fatores latentes dos usuários (usuários x k)
    - H: matriz de fatores latentes dos filmes (k x filmes)
    - losses: vetor com os valores de perda em cada iteração
    """
    n_users, n_movies = matrix.shape
    Y = matrix.copy()

    # Inicializa W e H com valores pequenos não-negativos
    #np.random.seed(42)
    W = np.random.rand(n_users, k) * 0.1
    H = np.random.rand(k, n_movies) * 0.1

    W_old = W.copy()
    H_old = H.copy()

    # Normaliza as colunas de W para somar 1
    W = W / np.sum(W, axis=0, keepdims=True)
    H = H / np.sum(H, axis=1, keepdims=True)

    # Máscara para elementos não-zero na matriz original
    mask = (Y > 0).astype(float)

    # Vetor para armazenar os valores de perda
    losses = []

    # Iterações do algoritmo
    for iter in range(max_iter):
        # Calcula a perda atual (RMSE nos elementos observados)
        prediction = W @ H
        error = mask * (Y - prediction)
        loss = np.sqrt(np.sum(error**2) / np.sum(mask)) #RMSE com máscara
        losses.append(loss)

        # Atualização de H usando a fórmula regularizada
        numerator_H = (W.T @ Y) - beta * H
        denominator_H = (W.T @ W @ H) + epsilon
        H = H * numerator_H / denominator_H

        # Atualização de W usando a fórmula regularizada
        numerator_W = (Y @ H.T) - alpha * W
        denominator_W = (W @ H @ H.T) + epsilon
        W = W * numerator_W / denominator_W

        # Normaliza colunas de W para somar 1
        W = W / np.sum(W, axis=0, keepdims=True)

        if H.sum() > 50*k*n_movies:
            print(f"H é muito grande, normalizando. Considere diminuir o valor de beta.")
            epsilon = 1e-5
            H = H / np.sum(H, axis=1, keepdims=True)

        # Verifica convergência
        W_change = np.max(W - W_old) / (np.max(W_old))
        H_change = np.max(H - H_old) / (np.max(H_old))

        if W_change < tol and H_change < tol:
            print(f"Convergência alcançada após {iter+1} iterações")
            break

        W_old = W.copy()
        H_old = H.copy()

        # Print para debugar
        #print(f"Soma total de H: {H.sum()}")

    if iter == max_iter - 1:
        print(f"Número máximo de iterações ({max_iter}) alcançado sem convergência")

    return W, H, losses

# %%
# Executar o algoritmo NMF no conjunto de treinamento
k = 10
W, H, losses = nmf(train_matrix, k, alpha=0.001, beta=0.0001)

# %%
# Visualizar a curva de perda
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Iteração')
plt.ylabel('Perda (RMSE)')
plt.title('Curva de Perda do NMF')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Gerar previsões usando o modelo NMF
predictions = W @ H

# Comparar previsões com valores reais no conjunto de treinamento
print("\nComparação de previsões no conjunto de treinamento:")

# Selecionar 10 pontos aleatórios com avaliações > 0
nonzero_indices = np.where(train_matrix > 0)
random_indices = np.random.choice(len(nonzero_indices[0]), 10, replace=False)
print("\nPontos com avaliações > 0:")
for i in random_indices:
    user = nonzero_indices[0][i]
    movie = nonzero_indices[1][i]
    real = train_matrix[user, movie]
    pred = predictions[user, movie]
    print(f"Usuário {user}, Filme {movie}: Real = {real:.2f}, Previsto = {pred:.2f}, Erro = {abs(real-pred):.2f}")

# Selecionar 10 pontos aleatórios com avaliações = 0
zero_indices = np.where(train_matrix == 0)
random_indices = np.random.choice(len(zero_indices[0]), 10, replace=False)
print("\nPontos com avaliações = 0:")
for i in random_indices:
    user = zero_indices[0][i]
    movie = zero_indices[1][i]
    pred = predictions[user, movie]
    print(f"Usuário {user}, Filme {movie}: Previsto = {pred:.2f}")

# %%
# Comparar previsões com valores reais no conjunto de validação
print("\nComparação de previsões no conjunto de validação:")

# Selecionar 10 pontos aleatórios com avaliações > 0
nonzero_indices = np.where(val_matrix > 0)
random_indices = np.random.choice(len(nonzero_indices[0]), 10, replace=False)
print("\nPontos com avaliações > 0:")
for i in random_indices:
    user = nonzero_indices[0][i]
    movie = nonzero_indices[1][i]
    real = val_matrix[user, movie]
    pred = predictions[user, movie]
    print(f"Usuário {user}, Filme {movie}: Real = {real:.2f}, Previsto = {pred:.2f}, Erro = {abs(real-pred):.2f}")

# %%
# Comparar previsões com valores reais no conjunto de teste
print("\nComparação de previsões no conjunto de teste:")

# Selecionar 10 pontos aleatórios com avaliações > 0
nonzero_indices = np.where(test_matrix > 0)
random_indices = np.random.choice(len(nonzero_indices[0]), 10, replace=False)
print("\nPontos com avaliações > 0:")
for i in random_indices:
    user = nonzero_indices[0][i]
    movie = nonzero_indices[1][i]
    real = test_matrix[user, movie]
    pred = predictions[user, movie]
    print(f"Usuário {user}, Filme {movie}: Real = {real:.2f}, Previsto = {pred:.2f}, Erro = {abs(real-pred):.2f}")
# %% [markdown]
# ## 3. Implementação do Sistema de Recomendação usando SLIM

# %%
def slim(matrix, learning_rate=0.01, l1_reg=0.01, l2_reg=0.01, max_iter=1000, tol=0.01, epsilon=1e-8):
    """
    Implementa o algoritmo SLIM (Sparse Linear Methods) usando gradiente descendente com projeção.

    Parâmetros:
    - matrix: matriz de treino R (usuários x itens)
    - learning_rate: taxa de aprendizado para o gradiente descendente
    - l1_reg: parâmetro de regularização L1 (controla a esparsidade)
    - l2_reg: parâmetro de regularização L2 (evita overfitting)
    - max_iter: número máximo de iterações
    - tol: tolerância para convergência (mudança relativa)
    - epsilon: pequeno valor para evitar divisão por zero

    Retorna:
    - M: matriz de similaridade entre itens (itens x itens)
    - losses: vetor com os valores de perda em cada iteração
    """
    n_users, n_items = matrix.shape
    R = matrix.copy()  # Matriz de ratings

    # Inicializar M com valores pequenos não-negativos
    np.random.seed(42)
    M = np.random.rand(n_items, n_items) * 0.01

    # Forçar diagonal para ser zero (um item não deve recomendar a si mesmo)
    np.fill_diagonal(M, 0)

    # Vetor para armazenar os valores de perda
    losses = []

    # Máscara para elementos não-zero na matriz original
    mask = (R > 0).astype(float)

    # Matriz M anterior para verificar convergência
    M_old = M.copy()

    # Iterações do algoritmo
    for iter in range(max_iter):
        # Calcular previsões: R_pred = R * M
        R_pred = R @ M

        # Calcular erro nos elementos observados
        error = mask * (R - R_pred)

        # Calcular perda (RMSE + regularização)
        rmse = np.sqrt(np.sum(error**2) / np.sum(mask))
        l1_penalty = l1_reg * np.sum(np.abs(M))
        l2_penalty = l2_reg * np.sum(M**2)
        loss = rmse + l1_penalty + l2_penalty
        losses.append(loss)

        # Calcular gradiente
        gradient = -2 * (R.T @ error) + 2 * l2_reg * M

        # Para regularização L1, adicionamos o sinal de M
        l1_grad = l1_reg * np.sign(M)
        gradient += l1_grad

        # Atualizar M usando gradiente descendente
        M = M - learning_rate * gradient

        # Projeção para garantir não-negatividade
        M = np.maximum(M, 0)

        # Forçar diagonal para ser zero
        np.fill_diagonal(M, 0)

        # Verificar convergência
        if np.max(M_old) > epsilon:
            rel_change = np.max(np.abs(M - M_old)) / np.max(np.abs(M_old))
            if rel_change < tol:
                print(f"Convergência alcançada após {iter+1} iterações")
                break

        M_old = M.copy()

        # Print para debugar (a cada 100 iterações)
        if iter % 100 == 0:
            print(f"Iteração {iter}, Perda: {loss:.4f}, RMSE: {rmse:.4f}")

    if iter == max_iter - 1:
        print(f"Número máximo de iterações ({max_iter}) alcançado sem convergência")

    return M, losses

# %%
# Executar o algoritmo SLIM no conjunto de treinamento
M, slim_losses = slim(train_matrix, learning_rate=0.001, l1_reg=0.1, l2_reg=0.1)

# %%
# Visualizar a curva de perda do SLIM
plt.figure(figsize=(10, 5))
plt.plot(slim_losses)
plt.xlabel('Iteração')
plt.ylabel('Perda (RMSE + Regularização)')
plt.title('Curva de Perda do SLIM')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Gerar previsões usando o modelo SLIM
slim_predictions = train_matrix @ M

# Comparar previsões com valores reais no conjunto de validação
print("\nComparação de previsões SLIM no conjunto de validação:")

# Selecionar 10 pontos aleatórios com avaliações > 0
nonzero_indices = np.where(val_matrix > 0)
random_indices = np.random.choice(len(nonzero_indices[0]), 10, replace=False)
print("\nPontos com avaliações > 0:")
val_errors = []
for i in random_indices:
    user = nonzero_indices[0][i]
    movie = nonzero_indices[1][i]
    real = val_matrix[user, movie]
    pred = slim_predictions[user, movie]
    error = abs(real - pred)
    val_errors.append(error)
    print(f"Usuário {user}, Filme {movie}: Real = {real:.2f}, Previsto = {pred:.2f}, Erro = {error:.2f}")

print(f"\nErro médio absoluto na validação: {np.mean(val_errors):.4f}")

# %%
# Comparar previsões com valores reais no conjunto de teste
print("\nComparação de previsões SLIM no conjunto de teste:")

# Selecionar 10 pontos aleatórios com avaliações > 0
nonzero_indices = np.where(test_matrix > 0)
random_indices = np.random.choice(len(nonzero_indices[0]), 10, replace=False)
print("\nPontos com avaliações > 0:")
test_errors = []
for i in random_indices:
    user = nonzero_indices[0][i]
    movie = nonzero_indices[1][i]
    real = test_matrix[user, movie]
    pred = slim_predictions[user, movie]
    error = abs(real - pred)
    test_errors.append(error)
    print(f"Usuário {user}, Filme {movie}: Real = {real:.2f}, Previsto = {pred:.2f}, Erro = {error:.2f}")

print(f"\nErro médio absoluto no teste: {np.mean(test_errors):.4f}")

# %%
# Visualizar a matriz de similaridade M
plt.figure(figsize=(10, 8))
plt.imshow(M, cmap='viridis')
plt.colorbar(label='Similaridade')
plt.title('Matriz de Similaridade SLIM')
plt.xlabel('Itens')
plt.ylabel('Itens')
plt.tight_layout()
plt.show()

# %%
# Calcular esparsidade da matriz M
sparsity = 1 - (np.count_nonzero(M) / (M.shape[0] * M.shape[1]))
print(f"Esparsidade da matriz M: {sparsity:.4f} ({sparsity*100:.2f}%)")

# %%
# Mostrar as top recomendações para alguns usuários aleatórios usando SLIM
def show_slim_recommendations(user_idx, n_recommendations=5):
    """
    Mostra as top recomendações para um usuário específico usando SLIM
    """
    # Filmes não avaliados pelo usuário
    unrated_movies = np.where(train_matrix[user_idx] == 0)[0]

    # Ordenar por pontuação prevista
    top_indices = unrated_movies[np.argsort(-slim_predictions[user_idx, unrated_movies])][:n_recommendations]

    print(f"\nTop {n_recommendations} recomendações SLIM para o Usuário {user_idx}:")
    for i, movie_idx in enumerate(top_indices, 1):
        score = slim_predictions[user_idx, movie_idx]
        print(f"{i}. Filme {movie_idx}: Pontuação prevista = {score:.2f}")

# Mostrar recomendações para 3 usuários aleatórios
random_users = np.random.choice(train_matrix.shape[0], 3, replace=False)
for user in random_users:
    show_slim_recommendations(user)
