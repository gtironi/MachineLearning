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
# %% [markdown]
# ## 4. Implementação do Sistema de Recomendação usando SLIM

# %%
def slim(matrix, learning_rate=0.001, l2_reg=0.001, l1_reg=0.0001, max_iter=1000, tol=0.01):
    """
    Implementa o algoritmo SLIM com gradiente descendente e projeção.

    Parâmetros:
    - R: matriz de treino (usuários x itens)
    - learning_rate: taxa de aprendizado
    - l2_reg: regularização L2
    - l1_reg: regularização L1
    - max_iter: número máximo de iterações
    - tol: tolerância para convergência

    Retorna:
    - W: matriz de similaridade entre itens
    - losses: valores de perda em cada iteração
    """
    R = matrix.copy()
    n_items = R.shape[1]

    mask = (R > 0).astype(int)

    # Inicialização com valores pequenos não-negativos
    W = np.random.rand(n_items, n_items) * 0.01
    np.fill_diagonal(W, 0)  # Garantir que Wii = 0

    losses = []

    # Iterações do algoritmo
    for iter in range(max_iter):
        W_old = W.copy()

        # Erro de reconstrução
        error = mask * (R - R @ W)

        # Gradiente com regularização
        gradient = -R.T @ error + l2_reg * W + l1_reg * np.sign(W)

        # Atualização com gradiente descendente
        W -= learning_rate * gradient

        # Projeção
        W[W < 0] = 0
        np.fill_diagonal(W, 0)

        # Calcula RMSE
        error = mask * (R - R @ W)
        loss = np.sqrt(np.sum(error**2) / np.sum(mask))
        print(loss)
        losses.append(loss)

        # Verificação de convergência
        change = np.max(np.abs(W - W_old)) / (np.max(W_old))
        if change < tol:
            print(f"Convergência alcançada após {iter+1} iterações")
            break

    if iter == max_iter - 1:
        print(f"Número máximo de iterações ({max_iter}) alcançado sem convergência")

    return W, losses

#%%
def slim_cord_desc(matrix, learning_rate=0.001, l2_reg=0.001, l1_reg=0.0001, max_iter=1000, tol=0.01):
    """
    Implementa o algoritmo SLIM com descida de coordenada e projeção.

    Parâmetros:
    - R: matriz de treino (usuários x itens)
    - learning_rate: taxa de aprendizado (não usada diretamente na Descida Coordenada simples, mas mantida para assinatura da função)
    - l2_reg: regularização L2
    - l1_reg: regularização L1
    - max_iter: número máximo de iterações
    - tol: tolerância para convergência

    Retorna:
    - W: matriz de similaridade entre itens
    - losses: valores de perda em cada iteração
    """
    R = matrix.copy()
    n_users, n_items = R.shape

    mask = (R > 0).astype(int)

    # Inicialização com valores pequenos não-negativos
    W = np.random.rand(n_items, n_items) * 0.01
    np.fill_diagonal(W, 0)  # Garantir que Wii = 0

    # Pré-computar as normas L2 ao quadrado das colunas de itens da matriz R
    # (equivalente a 'cnorms' na implementação em C)
    R_col_norms_sq = np.sum(R**2, axis=0)

    # Inicializar R_hat = R @ W. Esta matriz será atualizada incrementalmente.
    R_hat = R @ W

    losses = []

    # Iterações principais do algoritmo
    for iter_main in range(max_iter):
        W_old = W.copy() # Copia W para verificar a convergência no final da iteração

        # Loop sobre cada item 'j' que será o item alvo (coluna 'j' em W)
        for j in range(n_items):
            # Obtém a coluna alvo R_j (equivalente a 'y' no código C)
            target_column_Rj = R[:, j]

            # Loop sobre cada item 'k' que contribuirá para a predição do item 'j' (coluna 'k' em W_j)
            for k in range(n_items):
                # Restrição: Wii = 0 (um item não é similar a si mesmo)
                if j == k:
                    continue

                # Obtém o valor atual de W_jk
                old_W_jk = W[j, k]

                # Denominador: ||R_k||^2 + λ₂
                # Evita divisão por zero se uma coluna de R for toda zero
                if R_col_norms_sq[k] == 0:
                    continue
                denominator_val = R_col_norms_sq[k] + l2_reg

                # Calcula o termo do 'numerador', equivalente a `aTy - ip` no código C
                # `aTy`: Produto escalar da coluna alvo R_j e da coluna contribuinte R_k
                term_aTy = np.dot(R[:, k], target_column_Rj)

                # `ip`: Produto escalar da coluna contribuinte R_k e da predição atual para R_j
                #      (APÓS remover a contribuição de old_W_jk de R_hat[:, j])
                # Esta etapa é crucial para a Descida Coordenada, pois otimiza incrementalmente.
                y_hat_without_old_W_jk = R_hat[:, j] - R[:, k] * old_W_jk
                term_ip = np.dot(R[:, k], y_hat_without_old_W_jk)

                # O `numerador` final usado na fórmula de soft-thresholding do código C:
                numerator_val = term_aTy - term_ip

                # Calcula o novo W_jk usando soft-thresholding com restrição de não-negatividade
                # (A lógica é equivalente à condição `numerator > l1r ? (numerator - l1r) / denom : 0.0` do C)
                if numerator_val > l1_reg:
                    new_W_jk = (numerator_val - l1_reg) / denominator_val
                else:
                    new_W_jk = 0.0

                # Atualiza W e R_hat incrementalmente se W_jk mudou
                if new_W_jk != old_W_jk:
                    W[j, k] = new_W_jk
                    # Atualiza R_hat[:, j] somando a mudança na contribuição de R_k
                    R_hat[:, j] += R[:, k] * (new_W_jk - old_W_jk)

        # Calcula o RMSE para o W atual após iterar por todos os elementos
        # Apenas as entradas observadas são consideradas para o cálculo da perda
        current_error_matrix = mask * (R - R_hat)
        loss = np.sqrt(np.sum(current_error_matrix**2) / np.sum(mask))
        print(loss) # Imprime a perda como no código original
        losses.append(loss)

        # Verificação de convergência (mantendo a lógica do usuário, mas mais robusta)
        max_W_old_abs = np.max(np.abs(W_old))
        if max_W_old_abs > 1e-9: # Evita divisão por zero ou números muito pequenos
            change = np.max(np.abs(W - W_old)) / max_W_old_abs
        else: # Se W_old for quase zero, a "mudança" é a própria magnitude de W-W_old
            change = np.max(np.abs(W - W_old))

        if change < tol:
            print(f"Convergência alcançada após {iter_main+1} iterações")
            break

    if iter_main == max_iter - 1:
        print(f"Número máximo de iterações ({max_iter}) alcançado sem convergência")

    return W, losses
# %%
# Executar o algoritmo SLIM no conjunto de treinamento
learning_rate = 0.0001  # Reduzido para evitar instabilidade
l2_reg = 0.01
l1_reg = 0.01
max_iter = 10

W, losses_slim = slim(train_matrix, learning_rate, l2_reg, l1_reg, max_iter)

# %%
# Visualizar a curva de perda
plt.figure(figsize=(10, 5))
plt.plot(losses_slim)
plt.xlabel('Iteração')
plt.ylabel('Perda')
plt.title('Curva de Perda do SLIM')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Gerar previsões usando o modelo SLIM
predictions_slim = train_matrix @ W

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
    pred = predictions_slim[user, movie]
    print(f"Usuário {user}, Filme {movie}: Real = {real:.2f}, Previsto = {pred:.2f}, Erro = {abs(real-pred):.2f}")

# Selecionar 10 pontos aleatórios com avaliações = 0
zero_indices = np.where(train_matrix == 0)
random_indices = np.random.choice(len(zero_indices[0]), 10, replace=False)
print("\nPontos com avaliações = 0:")
for i in random_indices:
    user = zero_indices[0][i]
    movie = zero_indices[1][i]
    pred = predictions_slim[user, movie]
    print(f"Usuário {user}, Filme {movie}: Previsto = {pred:.2f}")

# %%
# Comparar previsões com valores reais no conjunto de validação
print("\nComparação de previsões SLIM no conjunto de validação:")

# Selecionar 10 pontos aleatórios com avaliações > 0
nonzero_indices = np.where(val_matrix > 0)
random_indices = np.random.choice(len(nonzero_indices[0]), 10, replace=False)
print("\nPontos com avaliações > 0:")
for i in random_indices:
    user = nonzero_indices[0][i]
    movie = nonzero_indices[1][i]
    real = val_matrix[user, movie]
    pred = predictions_slim[user, movie]
    print(f"Usuário {user}, Filme {movie}: Real = {real:.2f}, Previsto = {pred:.2f}, Erro = {abs(real-pred):.2f}")

# %%
# Comparar previsões com valores reais no conjunto de teste
print("\nComparação de previsões SLIM no conjunto de teste:")

# Selecionar 10 pontos aleatórios com avaliações > 0
nonzero_indices = np.where(test_matrix > 0)
random_indices = np.random.choice(len(nonzero_indices[0]), 10, replace=False)
print("\nPontos com avaliações > 0:")
for i in random_indices:
    user = nonzero_indices[0][i]
    movie = nonzero_indices[1][i]
    real = test_matrix[user, movie]
    pred = predictions_slim[user, movie]
    print(f"Usuário {user}, Filme {movie}: Real = {real:.2f}, Previsto = {pred:.2f}, Erro = {abs(real-pred):.2f}")

# %%
# Calcular RMSE no conjunto de validação
val_mask = (val_matrix > 0).astype(int)
val_error_slim = val_mask * (val_matrix - predictions_slim)
val_rmse_slim = np.sqrt(np.sum(val_error_slim**2) / np.sum(val_mask))
print(f"\nRMSE do SLIM no conjunto de validação: {val_rmse_slim:.4f}")

# %%
# %% [markdown]
# ## 5. Funções de Avaliação

# %%
def calculate_rmse(true_matrix, pred_matrix):
    """
    Calcula RMSE apenas nas posições onde há avaliações conhecidas
    """
    mask = (true_matrix > 0).astype(int)
    error = mask * (true_matrix - pred_matrix)
    rmse = np.sqrt(np.sum(error**2) / np.sum(mask))
    return rmse

def calculate_recall_at_k(true_matrix, pred_matrix, exclude_matrix=None, k=10, threshold=4):
    """
    Calcula Recall@k para cada usuário e retorna a média
    Considera apenas filmes com rating >= threshold como relevantes
    exclude_matrix: matriz adicional para excluir (ex: validação quando calculando teste)
    """
    n_users, n_movies = true_matrix.shape
    recalls = []

    for user in range(n_users):
        # Encontrar filmes com rating alto no conjunto atual
        relevant_movies = np.where(true_matrix[user] >= threshold)[0]

        # Se o usuário não tem filmes com rating alto, pula
        if len(relevant_movies) == 0:
            continue

        # Encontrar top-k filmes recomendados (maiores predições)
        pred_user = pred_matrix[user].copy()

        # Excluir filmes já avaliados no treino
        train_movies = np.where(train_matrix[user] > 0)[0]
        pred_user[train_movies] = -np.inf

        # Excluir filmes da matriz adicional (ex: validação quando avaliando teste)
        if exclude_matrix is not None:
            exclude_movies = np.where(exclude_matrix[user] > 0)[0]
            pred_user[exclude_movies] = -np.inf

        top_k_movies = np.argsort(pred_user)[-k:][::-1]

        # Calcular recall
        relevant_in_top_k = len(np.intersect1d(relevant_movies, top_k_movies))
        recall = relevant_in_top_k / len(relevant_movies)
        recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0

# %%
# %% [markdown]
# ## 6. Avaliação e Seleção de Hiperparâmetros

# %%
def evaluate_nmf_hyperparams(train_matrix, val_matrix):
    """
    Avalia diferentes hiperparâmetros para NMF usando conjunto de validação
    Retorna melhores parâmetros para RMSE e Recall separadamente
    """
    k_values = [5, 10, 15, 20]
    alpha_values = [0.0001, 0.001, 0.01]
    beta_values = [0.0001, 0.001, 0.01]

    best_rmse = float('inf')
    best_recall = 0.0
    best_params_rmse = {}
    best_params_recall = {}
    results = []

    print("Avaliando hiperparâmetros do NMF...")

    for k in k_values:
        for alpha in alpha_values:
            for beta in beta_values:
                print(f"Testando k={k}, alpha={alpha}, beta={beta}")

                # Treinar modelo
                W, H, losses = nmf(train_matrix, k, max_iter=200, alpha=alpha, beta=beta)
                predictions = W @ H

                # Avaliar no conjunto de validação
                rmse = calculate_rmse(val_matrix, predictions)
                recall = calculate_recall_at_k(val_matrix, predictions)

                results.append({
                    'k': k, 'alpha': alpha, 'beta': beta,
                    'rmse': rmse, 'recall': recall
                })

                # Atualizar melhores parâmetros para RMSE
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params_rmse = {'k': k, 'alpha': alpha, 'beta': beta}

                # Atualizar melhores parâmetros para Recall
                if recall > best_recall:
                    best_recall = recall
                    best_params_recall = {'k': k, 'alpha': alpha, 'beta': beta}

                print(f"RMSE: {rmse:.4f}, Recall@10: {recall:.4f}")

    print(f"\nMelhores parâmetros NMF (RMSE): {best_params_rmse} (RMSE: {best_rmse:.4f})")
    print(f"Melhores parâmetros NMF (Recall): {best_params_recall} (Recall: {best_recall:.4f})")
    return best_params_rmse, best_params_recall, results

def evaluate_svd_hyperparams(train_matrix, val_matrix):
    """
    Avalia diferentes hiperparâmetros para SVD usando conjunto de validação
    Retorna melhores parâmetros para RMSE e Recall@10 separadamente
    """
    k_values = [5, 10, 15, 20]
    reg_values = [0.0, 0.001, 0.01, 0.1]

    best_rmse = float('inf')
    best_recall = 0.0
    best_params_rmse = {}
    best_params_recall = {}
    results = []

    print("Avaliando hiperparâmetros do SVD...")

    for k in k_values:
        for reg in reg_values:
            print(f"Testando k={k}, regularization={reg}")

            # Treinar modelo
            Q, Sigma, P, losses = svd(train_matrix, k, max_iter=100, regularization=reg)
            predictions = Q @ Sigma @ P.T

            # Avaliar no conjunto de validação
            rmse = calculate_rmse(val_matrix, predictions)
            recall = calculate_recall_at_k(val_matrix, predictions)

            results.append({
                'k': k, 'regularization': reg,
                'rmse': rmse, 'recall': recall
            })

            # Atualizar melhores parâmetros para RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                best_params_rmse = {'k': k, 'regularization': reg}

            # Atualizar melhores parâmetros para Recall
            if recall > best_recall:
                best_recall = recall
                best_params_recall = {'k': k, 'regularization': reg}

            print(f"RMSE: {rmse:.4f}, Recall@10: {recall:.4f}")

    print(f"\nMelhores parâmetros SVD (RMSE): {best_params_rmse} (RMSE: {best_rmse:.4f})")
    print(f"Melhores parâmetros SVD (Recall): {best_params_recall} (Recall: {best_recall:.4f})")
    return best_params_rmse, best_params_recall, results

def evaluate_slim_hyperparams(train_matrix, val_matrix):
    """
    Avalia diferentes hiperparâmetros para SLIM usando conjunto de validação
    Retorna melhores parâmetros para RMSE e Recall@10 separadamente
    """
    learning_rate = 0.000001  # Conforme solicitado
    l2_values = [0.001, 0.01, 0.1]
    l1_values = [0.001, 0.01, 0.1]

    best_rmse = float('inf')
    best_recall = 0.0
    best_params_rmse = {}
    best_params_recall = {}
    results = []

    print("Avaliando hiperparâmetros do SLIM...")

    for l2_reg in l2_values:
        for l1_reg in l1_values:
            print(f"Testando l2_reg={l2_reg}, l1_reg={l1_reg}")

            # Treinar modelo
            W, losses = slim(train_matrix, learning_rate, l2_reg, l1_reg, max_iter=50)
            predictions = train_matrix @ W

            # Avaliar no conjunto de validação
            rmse = calculate_rmse(val_matrix, predictions)
            recall = calculate_recall_at_k(val_matrix, predictions)

            results.append({
                'l2_reg': l2_reg, 'l1_reg': l1_reg,
                'rmse': rmse, 'recall': recall
            })

            # Atualizar melhores parâmetros para RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                best_params_rmse = {'l2_reg': l2_reg, 'l1_reg': l1_reg}

            # Atualizar melhores parâmetros para Recall
            if recall > best_recall:
                best_recall = recall
                best_params_recall = {'l2_reg': l2_reg, 'l1_reg': l1_reg}

            print(f"RMSE: {rmse:.4f}, Recall@10: {recall:.4f}")

    print(f"\nMelhores parâmetros SLIM (RMSE): {best_params_rmse} (RMSE: {best_rmse:.4f})")
    print(f"Melhores parâmetros SLIM (Recall): {best_params_recall} (Recall: {best_recall:.4f})")
    return best_params_rmse, best_params_recall, results

# %%
# Executar avaliação de hiperparâmetros
np.random.seed(42)  # Para reprodutibilidade

# Avaliar NMF
best_nmf_params_rmse, best_nmf_params_recall, nmf_results = evaluate_nmf_hyperparams(train_matrix, val_matrix)

# %%
# Avaliar SVD
best_svd_params_rmse, best_svd_params_recall, svd_results = evaluate_svd_hyperparams(train_matrix, val_matrix)

# %%
# Avaliar SLIM
best_slim_params_rmse, best_slim_params_recall, slim_results = evaluate_slim_hyperparams(train_matrix, val_matrix)

# %%
# %% [markdown]
# ## 7. Treinamento dos Modelos Finais com Melhores Hiperparâmetros

# %%
print("Treinando modelos finais com melhores hiperparâmetros...")

# Treinar NMF final - RMSE otimizado
print("Treinando NMF otimizado para RMSE...")
W_final_rmse, H_final_rmse, losses_nmf_final_rmse = nmf(
    train_matrix,
    best_nmf_params_rmse['k'],
    max_iter=500,
    alpha=best_nmf_params_rmse['alpha'],
    beta=best_nmf_params_rmse['beta']
)
predictions_nmf_final_rmse = W_final_rmse @ H_final_rmse

# Treinar NMF final - Recall otimizado
print("Treinando NMF otimizado para Recall@10...")
W_final_recall, H_final_recall, losses_nmf_final_recall = nmf(
    train_matrix,
    best_nmf_params_recall['k'],
    max_iter=500,
    alpha=best_nmf_params_recall['alpha'],
    beta=best_nmf_params_recall['beta']
)
predictions_nmf_final_recall = W_final_recall @ H_final_recall

# %%
# Treinar SVD final - RMSE otimizado
print("Treinando SVD otimizado para RMSE...")
Q_final_rmse, Sigma_final_rmse, P_final_rmse, losses_svd_final_rmse = svd(
    train_matrix,
    best_svd_params_rmse['k'],
    max_iter=200,
    regularization=best_svd_params_rmse['regularization']
)
predictions_svd_final_rmse = Q_final_rmse @ Sigma_final_rmse @ P_final_rmse.T

# Treinar SVD final - Recall otimizado
print("Treinando SVD otimizado para Recall@10...")
Q_final_recall, Sigma_final_recall, P_final_recall, losses_svd_final_recall = svd(
    train_matrix,
    best_svd_params_recall['k'],
    max_iter=200,
    regularization=best_svd_params_recall['regularization']
)
predictions_svd_final_recall = Q_final_recall @ Sigma_final_recall @ P_final_recall.T

# %%
# Treinar SLIM final - RMSE otimizado
print("Treinando SLIM otimizado para RMSE...")
W_slim_final_rmse, losses_slim_final_rmse = slim(
    train_matrix,
    learning_rate=0.000001,  # Conforme solicitado
    l2_reg=best_slim_params_rmse['l2_reg'],
    l1_reg=best_slim_params_rmse['l1_reg'],
    max_iter=100
)
predictions_slim_final_rmse = train_matrix @ W_slim_final_rmse

# Treinar SLIM final - Recall otimizado
print("Treinando SLIM otimizado para Recall@10...")
W_slim_final_recall, losses_slim_final_recall = slim(
    train_matrix,
    learning_rate=0.000001,  # Conforme solicitado
    l2_reg=best_slim_params_recall['l2_reg'],
    l1_reg=best_slim_params_recall['l1_reg'],
    max_iter=100
)
predictions_slim_final_recall = train_matrix @ W_slim_final_recall

# %%
# %% [markdown]
# ## 8. Avaliação Final no Conjunto de Teste

# %%
# Calcular métricas no conjunto de teste
print("Avaliação final no conjunto de teste:")

# Modelos otimizados para RMSE
rmse_nmf_test_rmse = calculate_rmse(test_matrix, predictions_nmf_final_rmse)
rmse_svd_test_rmse = calculate_rmse(test_matrix, predictions_svd_final_rmse)
rmse_slim_test_rmse = calculate_rmse(test_matrix, predictions_slim_final_rmse)

recall_nmf_test_rmse = calculate_recall_at_k(test_matrix, predictions_nmf_final_rmse, exclude_matrix=val_matrix)
recall_svd_test_rmse = calculate_recall_at_k(test_matrix, predictions_svd_final_rmse, exclude_matrix=val_matrix)
recall_slim_test_rmse = calculate_recall_at_k(test_matrix, predictions_slim_final_rmse, exclude_matrix=val_matrix)

# Modelos otimizados para Recall
rmse_nmf_test_recall = calculate_rmse(test_matrix, predictions_nmf_final_recall)
rmse_svd_test_recall = calculate_rmse(test_matrix, predictions_svd_final_recall)
rmse_slim_test_recall = calculate_rmse(test_matrix, predictions_slim_final_recall)

recall_nmf_test_recall = calculate_recall_at_k(test_matrix, predictions_nmf_final_recall, exclude_matrix=val_matrix)
recall_svd_test_recall = calculate_recall_at_k(test_matrix, predictions_svd_final_recall, exclude_matrix=val_matrix)
recall_slim_test_recall = calculate_recall_at_k(test_matrix, predictions_slim_final_recall, exclude_matrix=val_matrix)

print(f"\nResultados dos modelos otimizados para RMSE:")
print(f"NMF  - RMSE: {rmse_nmf_test_rmse:.4f}, Recall@10: {recall_nmf_test_rmse:.4f}")
print(f"SVD  - RMSE: {rmse_svd_test_rmse:.4f}, Recall@10: {recall_svd_test_rmse:.4f}")
print(f"SLIM - RMSE: {rmse_slim_test_rmse:.4f}, Recall@10: {recall_slim_test_rmse:.4f}")

print(f"\nResultados dos modelos otimizados para Recall@10:")
print(f"NMF  - RMSE: {rmse_nmf_test_recall:.4f}, Recall@10: {recall_nmf_test_recall:.4f}")
print(f"SVD  - RMSE: {rmse_svd_test_recall:.4f}, Recall@10: {recall_svd_test_recall:.4f}")
print(f"SLIM - RMSE: {rmse_slim_test_recall:.4f}, Recall@10: {recall_slim_test_recall:.4f}")

# %%
# %% [markdown]
# ## 9. Visualização dos Resultados

# %%
# Gráfico de barras comparando modelos otimizados para RMSE vs Recall
models = ['NMF', 'SVD', 'SLIM']
rmse_values_rmse = [rmse_nmf_test_rmse, rmse_svd_test_rmse, rmse_slim_test_rmse]
recall_values_recall = [recall_nmf_test_recall, recall_svd_test_recall, recall_slim_test_recall]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# RMSE dos modelos otimizados para RMSE
colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars1 = ax1.bar(models, rmse_values_rmse, color=colors1, alpha=0.8, edgecolor='black')
ax1.set_ylabel('RMSE')
ax1.set_title('RMSE - Modelos Otimizados para RMSE')
ax1.grid(True, alpha=0.3)

for bar, value in zip(bars1, rmse_values_rmse):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# Recall@10 dos modelos otimizados para Recall
bars2 = ax2.bar(models, recall_values_recall, color=colors1, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Recall@10')
ax2.set_title('Recall@10 - Modelos Otimizados para Recall@10')
ax2.grid(True, alpha=0.3)

for bar, value in zip(bars2, recall_values_recall):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Gráficos das curvas de convergência
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 15))

# NMF
ax1.plot(losses_nmf_final_rmse, color='#FF6B6B', linewidth=2)
ax1.set_xlabel('Iteração')
ax1.set_ylabel('RMSE')
ax1.set_title('NMF - Otimizado para RMSE')
ax1.grid(True, alpha=0.3)

ax2.plot(losses_nmf_final_recall, color='#FF6B6B', linewidth=2)
ax2.set_xlabel('Iteração')
ax2.set_ylabel('RMSE')
ax2.set_title('NMF - Otimizado para Recall@10')
ax2.grid(True, alpha=0.3)

# SVD
ax3.plot(losses_svd_final_rmse, color='#4ECDC4', linewidth=2)
ax3.set_xlabel('Iteração')
ax3.set_ylabel('RMSE')
ax3.set_title('SVD - Otimizado para RMSE')
ax3.grid(True, alpha=0.3)

ax4.plot(losses_svd_final_recall, color='#4ECDC4', linewidth=2)
ax4.set_xlabel('Iteração')
ax4.set_ylabel('RMSE')
ax4.set_title('SVD - Otimizado para Recall@10')
ax4.grid(True, alpha=0.3)

# SLIM
ax5.plot(losses_slim_final_rmse, color='#45B7D1', linewidth=2)
ax5.set_xlabel('Iteração')
ax5.set_ylabel('RMSE')
ax5.set_title('SLIM - Otimizado para RMSE')
ax5.grid(True, alpha=0.3)

ax6.plot(losses_slim_final_recall, color='#45B7D1', linewidth=2)
ax6.set_xlabel('Iteração')
ax6.set_ylabel('RMSE')
ax6.set_title('SLIM - Otimizado para Recall@10')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Tabela resumo completa dos resultados
print("\n" + "="*100)
print("RESUMO DOS RESULTADOS FINAIS")
print("="*100)
print(f"{'Modelo':<15} {'Otimização':<12} {'RMSE Teste':<12} {'Recall@10':<12} {'Hiperparâmetros'}")
print("-"*100)

# Modelos otimizados para RMSE
print(f"{'NMF':<15} {'RMSE':<12} {rmse_nmf_test_rmse:<12.4f} {recall_nmf_test_rmse:<12.4f} k={best_nmf_params_rmse['k']}, α={best_nmf_params_rmse['alpha']}, β={best_nmf_params_rmse['beta']}")
print(f"{'SVD':<15} {'RMSE':<12} {rmse_svd_test_rmse:<12.4f} {recall_svd_test_rmse:<12.4f} k={best_svd_params_rmse['k']}, reg={best_svd_params_rmse['regularization']}")
print(f"{'SLIM':<15} {'RMSE':<12} {rmse_slim_test_rmse:<12.4f} {recall_slim_test_rmse:<12.4f} l2={best_slim_params_rmse['l2_reg']}, l1={best_slim_params_rmse['l1_reg']}")

print()
# Modelos otimizados para Recall
print(f"{'NMF':<15} {'Recall@10':<12} {rmse_nmf_test_recall:<12.4f} {recall_nmf_test_recall:<12.4f} k={best_nmf_params_recall['k']}, α={best_nmf_params_recall['alpha']}, β={best_nmf_params_recall['beta']}")
print(f"{'SVD':<15} {'Recall@10':<12} {rmse_svd_test_recall:<12.4f} {recall_svd_test_recall:<12.4f} k={best_svd_params_recall['k']}, reg={best_svd_params_recall['regularization']}")
print(f"{'SLIM':<15} {'Recall@10':<12} {rmse_slim_test_recall:<12.4f} {recall_slim_test_recall:<12.4f} l2={best_slim_params_recall['l2_reg']}, l1={best_slim_params_recall['l1_reg']}")

print("="*100)

# Determinar melhores modelos
all_rmse_values = rmse_values_rmse
all_recall_values = recall_values_recall
all_model_names = [f'{m} (RMSE)' for m in models] + [f'{m} (Recall)' for m in models]

best_rmse_idx = np.argmin(all_rmse_values)
best_recall_idx = np.argmax(all_recall_values)

print(f"\nMelhor modelo por RMSE: {all_model_names[best_rmse_idx]} ({all_rmse_values[best_rmse_idx]:.4f})")
print(f"Melhor modelo por Recall@10: {all_model_names[best_recall_idx]} ({all_recall_values[best_recall_idx]:.4f})")

# %%
