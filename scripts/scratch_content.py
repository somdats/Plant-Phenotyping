@jit(nopython=True)
def viterbi_inner_loop_parallel(omega, prev, T, E, V, N, M, start, end):
    for t in prange(start, end):  # Use prange for parallel loop
        for j in range(N):
            probability = omega[t - 1] + np.log(T[:, j]) + np.log(E[j, V[t]])
            max_idx = np.argmax(probability)
            prev[t - 1, j] = max_idx

            # Compute the maximum probability outside the inner loop
            max_probability = np.max(probability)
            omega[t, j] = max_probability

def viterbi(V, T, E, StateNames):
    M = V.shape[0]
    N = T.shape[0]

    omega = np.zeros((M, N))
    omega[0, :] = np.log(E[:, V[0]])
    st = time.time()
    prev = np.zeros((M - 1, N))

    for t in range(1, M):
        for j in range(N):
            # Same as Forward Probability
            #test_e = E[j, V[t]]
            tt = V[t]
            val = E[j, V[t]]
            test_e_log = np.log(val)
            #test_t = np.log(T[:, j])
            #val_x = np.log(T[:, j]) + np.log(E[j, V[t]])
            probability = omega[t - 1] + np.log(T[:, j]) + np.log(E[j, V[t]])
            #print("probability:",probability)
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)

            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)

        # Path Array
        S_ = np.zeros(M)

        # Find the most probable last hidden state
        last_state = np.argmax(omega[M - 1, :])

        S_[0] = last_state
    end = time.time()
    print("viterbi took {} seconds".format(round(end - st, 3)))
    backtrack_index = 1
    for i in range(M - 2, -1, -1):
        S_[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    # Flip the path array since we were backtracking
    S_ = np.flip(S_, axis=0)

    # Convert numeric values to actual hidden states
    S = []
    for s in S_:
        S.append(StateNames[int(s)])

    return S

def viterbi_algorithm_sparse(V, T, E, StateNames, invalid_value):

    M = V.shape[0]
    N = T.shape[0]

    T_log = np.log(np.where(T >= 1e6, 1e-20, T))  # Use log probabilities to avoid underflow for invalid elements
    E_log = np.log(np.where(E >= 1e5, 1e-20, E))  # Use log probabilities for observations and handle invalid elements

    V_log = np.log(V + 1e-20)  # Use log probabilities for observations

    # Initialize the Viterbi matrix and backtrace matrix
    Viterbi = np.zeros((N, len(V)))
    backtrace = np.zeros((N, len(V)), dtype=int)

    # Initialize the first column of Viterbi matrix with the initial probabilities
    Viterbi[:, 0] = T_log[:, 0] + E_log[:, V[0]]

    # Perform the dynamic programming step for each time step
    for t in range(1, len(V)):
        temp = Viterbi[:, t - 1][:, np.newaxis] + T_log + E_log[:, V[t]][np.newaxis, :]
        Viterbi[:, t] = np.max(temp, axis=0) + V_log[t]
        backtrace[:, t] = np.argmax(temp, axis=0)

    # Backtrack to find the best path
    S = [np.argmax(Viterbi[:, -1])]
    for t in range(len(V) - 1, 0, -1):
        S.append(backtrace[S[-1], t])

    # Convert state indices to state names
    S = [StateNames[state_idx] for state_idx in reversed(S)]

    return S
def viterbi_parallel_sparse(V, T, E, StateNames):
    M = V.shape[0]
    N = T.shape[0]

    omega = np.zeros((M, N))
    omega[0, :] = np.log(E[:, V[0]])

    prev = np.zeros((M - 1, N))
    st = time.time()

    T[T >= 1e6] = 0  # Replace invalid values in T with 0
    E[E >= 1e5] = 0   # Replace invalid values in E with 0
    T_sparse = csc_matrix(T)
    E_sparse = csc_matrix(E)

    def compute_probabilities(t):
        T_dense = T_sparse.toarray()
        E_dense = E_sparse.toarray()
        probabilities = omega[t - 1] + np.log(T_dense[:, :]) + np.log(E_dense[:, V[t]])
        return probabilities

    with concurrent.futures.ThreadPoolExecutor() as executor:
        probabilities = list(executor.map(compute_probabilities, range(1, M)))

        for t in range(1, M):
            for j in range(N):
                probability = probabilities[t - 1][j]

                prev[t - 1, j] = np.argmax(probability)
                omega[t, j] = np.max(probability)

            S_ = np.zeros(M, dtype=int)
            last_state = np.argmax(omega[M - 1, :])
            S_[0] = last_state

        backtrack_index = 1
        for i in range(M - 2, -1, -1):
            S_[backtrack_index] = prev[i, int(last_state)]
            last_state = prev[i, int(last_state)]
            backtrack_index += 1

        S_ = np.flip(S_, axis=0)

        S = []
        for s in S_:
            S.append(StateNames[int(s)])
    et = time.time()
    print("viterbi took {} seconds".format(round(et - st, 3)))
    return S

def viterbi_sparse_with_invalid(V, T, E, StateNames):
    M = V.shape[0]
    N = T.shape[0]

    omega = np.zeros((M, N))
    omega[0, :] = np.log(E[:, V[0]])

    prev = np.zeros((M - 1, N), dtype=np.int32)

    T_invalid_mask = T <= 1e-5
    #T_data = np.where(T_invalid_mask, 0, T)
    T_sparse = csr_matrix(T)

    E_invalid_mask = E <= 1e-5
    #E_data = np.where(E_invalid_mask, 0, E)
    E_sparse = csr_matrix(E)
    start = time.time()
    for t in range(1, M):
        for j in range(N):
            #test_t = T_sparse[:, j]
            #t_log = np.log(T_sparse[:, j].toarray().flatten())
            #test_e = E_sparse[:, V[t]]
            #e_log = np.log(E_sparse[:, V[t]].toarray().flatten())
            st = time.time()
            #working val
            #val_x = np.log(T_sparse[:, j].data) + np.log(E_sparse[:, V[t]].data)

            #val_x = np.log(T_sparse[:, j].toarray().flatten() + 1e-20) + np.log(E_sparse[:, V[t]].toarray().flatten() + 1e-20)
            probabilities = omega[t - 1] + np.log(T_sparse[:, j].data) + np.log(E_sparse[:, V[t]].data)[j]
            ed = time.time()
            print("viterbi inner looping took {} seconds".format(round(ed - st, 3)))
            # probability = probabilities[j]
            # print("probability:", probability)
            prev[t - 1, j] = np.argmax(probabilities)
            omega[t, j] = np.max(probabilities)

        S_ = np.zeros(M, dtype=np.int32)
        last_state = np.argmax(omega[M - 1, :])
        S_[0] = last_state

    end = time.time()
    print("viterbi looping took {} seconds".format(round(end - start, 3)))
    backtrack_index = 1
    # while backtrack_index < M:
    #     S_[backtrack_index] = prev[backtrack_index - 1, last_state]
    #     last_state = prev[backtrack_index - 1, last_state]
    #     backtrack_index += 1
    for i in range(M - 2, -1, -1):
        S_[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    S_ = np.flip(S_, axis=0)
    S = [StateNames[int(s)] for s in S_]

    return S

def viterbi_sparse_with_gpu(V, T, E, StateNames):

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    M = V.shape[0]
    N = T.shape[0]

    omega = np.zeros((M, N))
    omega[0, :] = np.log(E[:, V[0]])

    prev = np.zeros((M - 1, N), dtype=np.int32)

    T_invalid_mask = T <= 1e-5
    #T_data = np.where(T_invalid_mask, 0, T)
    T_sparse = csr_matrix(T)

    E_invalid_mask = E <= 1e-5
    #E_data = np.where(E_invalid_mask, 0, E)
    E_sparse = csr_matrix(E)

    # Convert data to PyTorch tensors and move to GPU
    gpu_omega = torch.tensor(omega, dtype=torch.float32, device=device)
    gpu_T_sparse = torch.tensor(T_sparse.toarray(), dtype=torch.float32, device=device)
    gpu_E_sparse = torch.tensor(E_sparse.toarray(), dtype=torch.float32, device=device)

    start = time.time()

    for t in range(1, M):
        for j in range(N):
            log_omega_t_minus_1 = torch.log(gpu_omega[t - 1, :])
            log_T_sparse = torch.log(gpu_T_sparse[:,j])
            log_E_sparse = torch.log(gpu_E_sparse[:, V[t]])[j]
            log_probabilities = log_omega_t_minus_1 + log_T_sparse  + log_E_sparse


            # Calculate the max probability and argmax for each state
            # max_probabilities = torch.max(log_probabilities)
            # argmax_states = torch.argmax(log_probabilities)
            # st = time.time()
            # Update the omega matrix and prev array
            gpu_omega[t, j] = torch.max(log_probabilities)
            prev[t - 1, j] = torch.argmax(log_probabilities).cpu().numpy()
            # ed = time.time()
            # print("viterbi internal looping took {} seconds".format(round(ed - st, 3)))

        S_ = np.zeros(M, dtype=np.int32)
        st = time.time()
        last_state = np.argmax(gpu_omega[M - 1, :].cpu().numpy())
        ed = time.time()
        #print("viterbi internal looping took {} seconds".format(round(ed - st, 3)))
        S_[0] = last_state


    end = time.time()
    print("viterbi looping took {} seconds".format(round(end - start, 3)))

    backtrack_index = 1
    for i in range(M - 2, -1, -1):
        S_[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    S_ = np.flip(S_, axis=0)
    S = [StateNames[int(s)] for s in S_]

    return S

def viterbi_parallel(V, T, E, StateNames):
    M = V.shape[0]
    N = T.shape[0]

    omega = np.zeros((M, N))
    omega[0, :] = np.log(E[:, V[0]])

    prev = np.zeros((M - 1, N))

    def calculate_probabilities(t, j):
        probability = omega[t - 1] + np.log(T[:, j]) + np.log(E[j, V[t]])
        return np.argmax(probability), np.max(probability)

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for t in range(1, M):
            for j in range(N):
                futures.append(executor.submit(calculate_probabilities, t, j))

        for t in range(1, M):
            for j in range(N):
                index, max_prob = futures.pop(0).result()
                prev[t - 1, j] = index
                omega[t, j] = max_prob

        S_ = np.zeros(M)
        last_state = np.argmax(omega[M - 1, :])
        S_[0] = last_state

    end = time.time()
    print("viterbi looping took {} seconds".format(round(end - start, 3)))

    backtrack_index = 1
    for i in range(M - 2, -1, -1):
        S_[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    S_ = np.flip(S_, axis=0)
    S = [StateNames[int(s)] for s in S_]

    return S
