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

    # Normaliza as colunas de W para somar 1
    W = W / np.sum(W, axis=0, keepdims=True)
    H = H / np.sum(H, axis=1, keepdims=True)

    # Máscara para elementos não-zero na matriz original
    mask = (Y > 0).astype(float)

    # Vetor para armazenar os valores de perda
    losses = []

    # Iterações do algoritmo
    for iter in range(max_iter):
        W_old = W.copy()
        H_old = H.copy()

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
        W_change = np.max(np.abs(W - W_old)) / (np.max(W_old))
        H_change = np.max(np.abs(H - H_old)) / (np.max(H_old))

        if W_change < tol and H_change < tol:
            print(f"Convergência alcançada após {iter+1} iterações")
            break

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
# %%
# Calcular RMSE no conjunto de validação
val_mask = (val_matrix > 0).astype(int)
val_error = val_mask * (val_matrix - predictions)
val_rmse = np.sqrt(np.sum(val_error**2) / np.sum(val_mask))
print(f"\nRMSE no conjunto de validação: {val_rmse:.4f}")
# %%
# %% [markdown]
# ## 3. Implementação do Sistema de Recomendação usando SVD
# %%
def svd(matrix, k, max_iter=1000, tol=0.01, regularization=0.0):
    """
    Implementa o algoritmo de recomendação baseado em SVD para matrizes com valores ausentes.

    Parâmetros:
    - matrix: matriz de treino R (usuários x filmes) com valores ausentes (zeros)
    - k: número de fatores latentes (rank da aproximação)
    - max_iter: número máximo de iterações
    - tol: tolerância para convergência
    - regularization: parâmetro de regularização para SVD

    Retorna:
    - Q: matriz de fatores latentes dos usuários
    - Sigma: matriz diagonal de valores singulares
    - P: matriz de fatores latentes dos filmes
    - losses: vetor com os valores de perda em cada iteração
    """
    R = matrix.copy()
    n_users, n_movies = R.shape

    losses = []

    # Máscara para para ausentes e presentes
    missing_mask = (R == 0)
    mask = (R > 0).astype(int)

    # Inicializa Rf preenchendo valores ausentes com médias das linhas (vetorizado)
    Rf = R.copy()
    row_means = np.sum(R, axis=1) / np.sum(mask, axis=1)
    means_matrix = np.tile(row_means.reshape(-1, 1), (1, n_movies))
    Rf = R + (means_matrix * missing_mask.astype(int))

    # Iterações do algoritmo
    for iter in range(max_iter):
        Rf_old = Rf.copy()

        # SVD de rank-k em Rf
        if regularization > 0:
            # SVD regularizado
            U, s, Vt = np.linalg.svd(Rf, full_matrices=False)

            s_reg = s / (s + regularization)

            Q = U[:, :k]
            Sigma = np.diag(s[:k] * s_reg[:k])
            P = Vt[:k, :].T
        else:
            # SVD padrão
            U, s, Vt = np.linalg.svd(Rf, full_matrices=False)

            Q = U[:, :k]
            Sigma = np.diag(s[:k])
            P = Vt[:k, :].T

        R_approx = Q @ Sigma @ P.T

        # Atualizar apenas as entradas originalmente ausentes
        Rf = R.copy()
        Rf[missing_mask] = R_approx[missing_mask]

        # Calcula RMSE
        error = mask * (R - R_approx)
        loss = np.sqrt(np.sum(error**2) / np.sum(mask))
        losses.append(loss)

        # Verificar convergência
        change = np.max(np.abs(Rf - Rf_old)) / (np.max(Rf_old))
        if change < tol:
            print(f"Convergência alcançada após {iter+1} iterações")
            break

    if iter == max_iter - 1:
        print(f"Número máximo de iterações ({max_iter}) alcançado sem convergência")


    # Calcular a decomposição SVD final
    U, s, Vt = np.linalg.svd(Rf, full_matrices=False)
    Q = U[:, :k]
    Sigma = np.diag(s[:k])
    P = Vt[:k, :].T

    return Q, Sigma, P, losses
# %%
# Executar o algoritmo SVD no conjunto de treinamento
k = 10
Q, Sigma, P, losses_svd = svd(train_matrix, k, regularization=0)

# Visualizar a curva de perda
plt.figure(figsize=(10, 5))
plt.plot(losses_svd)
plt.xlabel('Iteração')
plt.ylabel('Perda (RMSE)')
plt.title('Curva de Perda do SVD')
plt.grid(True, alpha=0.3)
plt.show()
# %%
# Gerar previsões usando o modelo SVD
predictions_svd = Q @ Sigma @ P.T

print("\nComparação de previsões no conjunto de treinamento:")

# Selecionar 10 pontos aleatórios com avaliações > 0
nonzero_indices = np.where(train_matrix > 0)
random_indices = np.random.choice(len(nonzero_indices[0]), 10, replace=False)
print("\nPontos com avaliações > 0:")
for i in random_indices:
    user = nonzero_indices[0][i]
    movie = nonzero_indices[1][i]
    real = train_matrix[user, movie]
    pred = predictions_svd[user, movie]
    print(f"Usuário {user}, Filme {movie}: Real = {real:.2f}, Previsto = {pred:.2f}, Erro = {abs(real-pred):.2f}")

# Selecionar 10 pontos aleatórios com avaliações = 0
zero_indices = np.where(train_matrix == 0)
random_indices = np.random.choice(len(zero_indices[0]), 10, replace=False)
print("\nPontos com avaliações = 0:")
for i in random_indices:
    user = zero_indices[0][i]
    movie = zero_indices[1][i]
    pred = predictions_svd[user, movie]
    print(f"Usuário {user}, Filme {movie}: Previsto = {pred:.2f}")
# %%
# Comparar previsões com valores reais no conjunto de validação
print("\nComparação de previsões SVD no conjunto de validação:")

# Selecionar 10 pontos aleatórios com avaliações > 0
nonzero_indices = np.where(val_matrix > 0)
random_indices = np.random.choice(len(nonzero_indices[0]), 10, replace=False)
print("\nPontos com avaliações > 0:")
for i in random_indices:
    user = nonzero_indices[0][i]
    movie = nonzero_indices[1][i]
    real = val_matrix[user, movie]
    pred = predictions_svd[user, movie]
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
    pred = predictions_svd[user, movie]
    print(f"Usuário {user}, Filme {movie}: Real = {real:.2f}, Previsto = {pred:.2f}, Erro = {abs(real-pred):.2f}")
# %%
# Calcular RMSE no conjunto de validação
val_mask = (val_matrix > 0).astype(int)
val_error = val_mask * (val_matrix - predictions_svd)
val_rmse = np.sqrt(np.sum(val_error**2) / np.sum(val_mask))
print(f"\nRMSE no conjunto de validação: {val_rmse:.4f}")

# %%
