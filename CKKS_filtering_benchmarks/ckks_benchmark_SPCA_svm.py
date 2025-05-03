import tenseal as ts
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from numpy.polynomial import Chebyshev
import concurrent.futures
import multiprocessing

def create_ckks_context(levels=7):
    # Safe pre-defined param sets for different levels
    #poly_modulus_degree	Max Total coeff_modulus Bits
    #1024	                  27
    #2048	                  54
    #4096	                  109
    #8192	                  218
    #16384	                  438
    #32768	                  881

    param_sets = {
        "3": {
            "coeffs": [37, 30, 37],
            "degree": 4096
        },
        "5": {
            "coeffs": [50, 35, 35, 35, 50],
            "degree": 8192
        },
        "6": {
            "coeffs": [40, 35, 33, 33, 35, 40],
            "degree": 8192
        },
        "7": {
            "coeffs": [34, 30, 30, 30, 30, 30, 34],
            "degree": 8192
        },

        "8": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 50],
            "degree": 16384
        },
        "9": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 16384
        },
        "10": {
            "coeffs": [40, 35, 35, 35, 35, 35, 35, 35, 35, 40],
            "degree": 16384
        },

        "11": {
            "coeffs": [35, 32, 32, 32, 32, 32, 32, 32, 32, 32, 35],
            "degree": 16384
        },

        "12": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 32768
        },
        "13": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 32768
        },
        "14": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 32768
        },
        "15": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 32768
        },
        "16": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 32768
        },
        "17": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 32768
        },
        "17": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 32768
        },
        "18": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 32768
        },
        "19": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 32768
        },
        "20": {
            "coeffs": [50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50],
            "degree": 32768
        }
    }


    levels_str = str(levels)
    if levels_str not in param_sets:
        raise ValueError(f"❌ Unsupported level '{levels}'. Supported levels: {list(param_sets.keys())}")

    params = param_sets[levels_str]
    coeffs = params["coeffs"]
    degree = params["degree"]
    total_bits = sum(coeffs)

    print(f"⚙️ Checking context parameters... Degree: {degree}, Coefficients: {coeffs}, Total bits: {total_bits}")

    try:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=degree,
            coeff_mod_bit_sizes=coeffs,
        )
        context.global_scale = 2 ** 40
        context.generate_galois_keys()
        print(f"🔐 CKKS context created with {levels} levels, total {total_bits} bits, degree {degree}.")
        return context
    except ValueError as e:
        print(f"❌ Error while creating context: {e}")
        raise


# Step 2: Encrypt a vector using CKKS
def encrypt_vector(context, vector):
    return ts.ckks_vector(context, vector)


# Step 3: PCA projection (vector × matrix.T)
def pca_projection(plain_pca_matrix, enc_input_vector):
    return enc_input_vector.matmul(plain_pca_matrix)


def svm_inference(svm_weights, svm_biases, enc_proj_vector, svm_deg, context):
    """
    Compute the homomorphic sum of encrypted dot products with SVM weights.
    - svm_weights: list of NumPy arrays (each support vector)
    - svm_biases: list of floats (bias for each support vector)
    - enc_proj_vector: CKKSVector (encrypted input after PCA)
    - context: TenSEAL context (used for initializing the encrypted zero accumulator)
    """
    # Initialize encrypted accumulator to zero
    enc_score_sum = ts.ckks_vector(context, [0.0] * enc_proj_vector.size())

    for i in range(len(svm_weights)):
        weight = svm_weights[i]
        bias = svm_biases[i]

        # Compute encrypted dot product and add bias (in plaintext)
        score = enc_proj_vector.dot(weight) + bias

        # Accumulate homomorphically
        enc_score_sum += score.pow(svm_deg)

    return enc_score_sum  # Still encrypted



def parallel_svm_inference(svm_weights, svm_biases, enc_proj_vector, svm_deg, context):
    """
    Parallelize the homomorphic sum of encrypted dot products with SVM weights.
    - svm_weights: list of NumPy arrays (each support vector)
    - svm_biases: list of floats (bias for each support vector)
    - enc_proj_vector: CKKSVector (encrypted input after PCA)
    - context: TenSEAL context (used for initializing the encrypted zero accumulator)
    """
    # Initialize encrypted accumulator to zero
    enc_score_sum = ts.ckks_vector(context, [0.0] * 1)

    # Function to compute encrypted dot product and accumulate the result
    def compute_dot_product(i):
        weight = svm_weights[i]
        bias = svm_biases[i]

        # Compute encrypted dot product and add bias (in plaintext)
        score = enc_proj_vector.dot(weight) + bias
        return score.pow(svm_deg)

    # Use ThreadPoolExecutor to parallelize the loop
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the compute_dot_product function to each index in the range of svm_weights
        results = executor.map(compute_dot_product, range(len(svm_weights)))

    # Sum the results homomorphically
    for result in results:
        enc_score_sum += result

    return enc_score_sum  # Still encrypted

def batched_svm_inference(svm_weights, svm_biases, enc_proj_vector, svm_deg, context):
    """
    Efficient SVM inference using matrix-style batching.

    Args:
        svm_weights (List[np.ndarray]): List of support vectors (each of shape [proj_dim])
        svm_biases (List[float]): List of biases for each support vector
        enc_proj_vector (ts.CKKSVector): Encrypted input vector (shape: [proj_dim])
        context (ts.Context): TenSEAL context

    Returns:
        ts.CKKSVector: Encrypted scalar score (sum of all SVM activations)
    """
    # Convert weights list into a single NumPy matrix (shape: [num_svs, proj_dim])
    weight_matrix = np.vstack(svm_weights)  # shape: (num_svs, proj_dim)
    bias_vector = np.array(svm_biases)      # shape: (num_svs,)

    # NOTE: enc_proj_vector is encrypted, weight_matrix is plaintext
    enc_scores_vector = enc_proj_vector.matmul(weight_matrix.T)  # encrypted vector (Packed Halevi and Should method)

    # Add bias in plaintext (will be added to each corresponding slot)
    enc_scores_vector += bias_vector
    if svm_deg > 1 : #Avoid involking .pow() when degree is 1
        enc_scores_vector.pow(svm_deg)
    # Sum all slots to get final encrypted scalar decision score (TotalSum Halevi and Shoup)
    enc_score = enc_scores_vector.sum()

    return enc_score




def parallel_svm_inference_multiprocessing(svm_weights, svm_biases, enc_proj_vector, svm_deg, context):
    """
    Parallelize the homomorphic sum of encrypted dot products with SVM weights using multiprocessing.
    - svm_weights: list of NumPy arrays (each support vector)
    - svm_biases: list of floats (bias for each support vector)
    - enc_proj_vector: CKKSVector (encrypted input after PCA)
    - context: TenSEAL context (used for initializing the encrypted zero accumulator)
    """
    # Initialize encrypted accumulator to zero
    enc_score_sum = ts.ckks_vector(context, [0.0] * enc_proj_vector.size())

    def compute_dot_product(i):
        weight = svm_weights[i]
        bias = svm_biases[i]

        # Compute encrypted dot product and add bias (in plaintext)
        score = enc_proj_vector.dot(weight) + bias
        return score.pow(svm_deg)

    with multiprocessing.Pool() as pool:
        # Use pool.map for parallelization
        results = pool.map(compute_dot_product, range(len(svm_weights)))

    # Sum the results homomorphically
    for result in results:
        enc_score_sum += result

    return enc_score_sum  # Still encrypted



# Sigmoid function for reference (public version)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def chebyshev_interpolator(func, a, b, degree):
    """
    Computes the Chebyshev polynomial approximation of a given function func over the interval [a, b]

    :param func: function to approximate (float -> float)
    :param a: lower bound of interval
    :param b: upper bound of interval
    :param degree: degree of the Chebyshev polynomial
    :return: Chebyshev polynomial object
    """
    # Generate Chebyshev nodes (mapped to the interval [a, b])
    nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))

    # Evaluate the function at the Chebyshev nodes
    f_vals = func(nodes * (b - a) / 2 + (b + a) / 2)

    # Create the Chebyshev approximation
    cheb_poly = Chebyshev.fit(nodes * (b - a) / 2 + (b + a) / 2, f_vals, degree)

    # Print the Chebyshev coefficients
    return cheb_poly.coef



def plot_polynomial(coeffs, x_range=(-10, 10), num_points=1000):
    """
    Plot a polynomial and the sigmoid function over a specified interval.

    Parameters:
    - coeffs: List of floats representing the polynomial coefficients.
    - x_range: Tuple representing the interval (min, max) over which to plot the polynomial and sigmoid.
    - num_points: Number of points to use for plotting within the x_range.
    """
    # Create a numpy array of x values over the specified range
    x_vals = np.linspace(x_range[0], x_range[1], num_points)

    # Evaluate the polynomial at each x value
    y_poly =np.polynomial.chebyshev.chebval(x_vals, coeffs)
    # Evaluate the sigmoid function at each x value
    y_sigmoid = 1 / (1 + np.exp(-x_vals))

    # Plot the polynomial and the sigmoid function
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_poly, label=f"Polynomial", color='blue')
    plt.plot(x_vals, y_sigmoid, label="Sigmoid Function", color='red', linestyle='--')

    plt.title("Plot of Polynomial and Sigmoid Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

def get_ckksvector_level(enc_vec):
    """
    Determine the level of a CKKSVector by comparing its parms_id to the SEALContext chain.
    """
    # Get SEALContext
    seal_ctx = enc_vec.context().seal_context()

    # Extract the actual ciphertext (handles case where .ciphertext() returns a list)
    ct = enc_vec.ciphertext()
    if isinstance(ct, list):
        ct = ct[0]

    current_pid = ct.parms_id()

    # Walk the modulus chain to find the level
    ctx_data = seal_ctx.data.key_context_data()
    level = 0
    while ctx_data:
        if ctx_data.parms_id() == current_pid:
            return level
        ctx_data = ctx_data.next_context_data()
        level += 1

    return -1  # Not found


def encrypted_inverse_newtonraphson(ctxt, context, iterations=5, initial_guess=0.5):
    """
    Approximates the inverse of an encrypted scalar using Newton-Raphson method.
    Args:
        ctxt: ts.CKKSVector – ciphertext containing the encrypted number `a`.
        context: ts.Context – the TenSEAL context used for encryption.
        iterations: int – number of Newton-Raphson iterations to run.
        initial_guess: float – initial guess for 1/a (should be reasonably close).
    Returns:
        ts.CKKSVector – encrypted approximation of 1/a.
    """
    # Encrypt the initial guess
    x = ts.ckks_vector(context, [initial_guess])  # Could also be a vector of same shape as ctxt
    for _ in range(iterations):
        # x = x * (2 - a * x)
        ax = ctxt * x
        two_minus_ax = 2 - ax
        x = x * two_minus_ax
    return x


def benchmark_encrypted_inverse_newtonraphson(levels=10, val=4.0, iterations=3, initial_guess=2):
    """
    Benchmark the homomorphic Newton-Raphson method for inverse approximation with TenSEAL.
    Args:
        levels: Number of CKKS levels available.
        val: The plaintext input to invert homomorphically.
        iterations: Number of Newton-Raphson iterations to run.
        initial_guess: Initial plaintext guess for 1/val.
    """
    assert levels >= iterations + 2, f"Too few levels ({levels}) for {iterations} Newton-Raphson iterations. Recommended >= {iterations + 2}"

    print("🏁 Running CKKS Encrypted Inverse Benchmark...")
    print(f"Input value: {val}, Iterations: {iterations}, Initial guess: {initial_guess}\n")

    # 1. Create TenSEAL CKKS context
    context = create_ckks_context(levels=levels)
    # 2. Encrypt input
    t0 = time.time()
    enc_val = ts.ckks_vector(context, [val])
    t1 = time.time()
    print(f"🔐 Input encryption time: {t1 - t0:.4f} s. Initial levels: {levels}\n")
    # 3. Homomorphic inverse using Newton-Raphson
    x = initial_guess
    print("🔁 Running Newton-Raphson iterations:")
    for i in range(iterations):
        start_iter = time.time()
        ax = enc_val * x
        two_minus_ax = 2 - ax
        x = x * two_minus_ax
        end_iter = time.time()
        lvl = get_ckksvector_level(x)
        print(f"   Iter {i+1}: time = {end_iter - start_iter:.4f} s, remaining levels = {levels - lvl}")

    # 4. Decrypt result
    t2 = time.time()
    approx_inv = x.decrypt()[0]
    t3 = time.time()

    print(f"\n✅ Encrypted inverse approximation: {approx_inv:.8f}")
    print(f"🔓 Decryption time: {t3 - t2:.4f} s")
    print(f"📏 Ground truth 1/{val}: {1.0 / val:.8f}")
    print(f"🧮 Absolute error: {abs((1.0 / val) - approx_inv):.8e}")
    print(f"🕒 Total time (including encryption, NR iterations, decryption): {t3 - t0:.4f} s\n")



def benchmark_secure_pca_svm(levels=10, input_dim=2000, proj_dim=64, num_svs=20, svm_deg=1, interpolation_degree=32):

    # Ensure the multiplicative depth (levels) is safely within the supported depth of the polynomial evaluator.
    # The square root of the interpolation degree is a loose estimate of the multiplicative depth supported.
    # Subtracting 2 gives a safety margin to avoid running out of levels.
    assert levels >= math.ceil(math.sqrt(interpolation_degree)) + 3, \
        f"Too few levels ({levels}) for interpolation degree {interpolation_degree}. Required {math.ceil(math.sqrt(interpolation_degree)) + 3}."


    print("🏁 Running CKKS Secure PCA + SVM Benchmark...")
    print(f"Input dim: {input_dim}, Projected dim: {proj_dim}, Num Support vectors: {num_svs}\n")

    # 1. Generate random PCA matrix and input
    plain_pca_matrix = np.random.randn(input_dim, proj_dim)
    svm_weights = [np.random.randn(proj_dim) for _ in range(num_svs)]
    svm_biases = [np.random.uniform(-1, 1) for _ in range(num_svs)]
    input_vector = np.random.randn(input_dim)

    context = create_ckks_context(levels=levels)

    # 3. Encrypt input vector
    t0 = time.time()
    enc_input = encrypt_vector(context, input_vector)
    t1 = time.time()
    level = get_ckksvector_level(enc_input)
    print(f"🔐 Input encryption time: {t1 - t0:.4f} s. Remaining levels {levels - level} ")

    # 4. PCA projection
    t2 = time.time()
    enc_projected = pca_projection(plain_pca_matrix, enc_input)
    t3 = time.time()
    level = get_ckksvector_level(enc_projected)
    print(f"📉 PCA projection time: {t3 - t2:.4f} s. Remaining levels {levels - level} ")

    # 5. SVM inference
    t4 = time.time()
    enc_score = batched_svm_inference(svm_weights, svm_biases, enc_projected, svm_deg, context)
    t5 = time.time()
    level = get_ckksvector_level(enc_score)
    print(f"📈 SVM inference time: {t5 - t4:.4f} s. Remaining levels {levels - level} ")
    #print(f'intermediate check of score values {enc_score.decrypt()}')
    chebyshev_sigmoid = chebyshev_interpolator(sigmoid, -20, 20, interpolation_degree)
    #plot_polynomial(chebyshev_sigmoid)
    t6 = time.time()
    decision = enc_score.polyval(chebyshev_sigmoid)
    t7 = time.time()
    level = get_ckksvector_level(decision)
    print(f"📈 Chebyshev Sigmoid Approximation Time: {t7 - t6:.4f} s. Remaining levels {levels - level}")
    t8 = time.time()
    pt_decision = decision.decrypt()
    t9 = time.time()

    print(f"Chebyshev Sigmoid Approximation : {pt_decision}")
    # 6. Decrypt results

    print(f"🔓 Decryption time: {t9 - t8:.4f} s\n")
    #print("✅ SVM decision:", pt_decision)
    print(f"Total filtering time : {t9 - t2:.4f} s\n")





if __name__ == "__main__":
    #benchmark_encrypted_inverse_newtonraphson()
    #exit()
    #Backdoor (Level 11 is sufficient)
    print('-----------------------------Benchmarking Property extraction for Backdoor filtering--------------------------------')
    benchmark_secure_pca_svm(levels=11, input_dim=320, proj_dim=8, num_svs=48, svm_deg=1, interpolation_degree=32)

    #Label flipping (Requires an extra level for the polynomial)
    print('\n\n\n-----------------------------Benchmarking Property extraction for Label-flipping filtering------------------------------')
    benchmark_secure_pca_svm(levels=12, input_dim=136, proj_dim=64, num_svs=76, svm_deg=2, interpolation_degree=32)

    #Gradient-Acsent
    print('\n\n\n-----------------------------Benchmarking Property extraction for Gradient-Ascent filtering------------------------------')
    benchmark_secure_pca_svm(levels=11, input_dim=320, proj_dim=8, num_svs=27, svm_deg=1, interpolation_degree=32)

    #Free-riders
    print('\n\n\n-----------------------------Benchmarking Property extraction for Free-riders filtering------------------------------')
    benchmark_secure_pca_svm(levels=12, input_dim=320, proj_dim=128, num_svs=97, svm_deg=2, interpolation_degree=32)
