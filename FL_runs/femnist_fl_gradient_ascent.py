import numpy as np
import csv
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, f1_score
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow.keras.models import clone_model
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from shadow_visualization import visualize_high_dimensional_data


#Local imports
from fl_utils import *
from SupervisedPCA import *



def test_on_dp_updates(svm, pca, layer, epsilon) :
    honest_weights = []
    dishonest_weights = []

    # Filepaths
    honest_filename = f"delta_shadow_updates_with_dp/GradAscent/Layer{str(layer)}/honest_{epsilon}_.csv"
    dishonest_filename = f"delta_shadow_updates_with_dp/GradAscent/Layer{str(layer)}/dishonest_{epsilon}_.csv"
    #visualize_high_dimensional_data(honest_filename, dishonest_filename)
    # Read data
    honest_data = np.loadtxt(honest_filename, delimiter=",")
    dishonest_data = np.loadtxt(dishonest_filename, delimiter=",")

    # Append data to respective lists
    honest_weights.append(1 * honest_data if honest_data.ndim > 1 else honest_data[np.newaxis, :])
    dishonest_weights.append(1 * dishonest_data if dishonest_data.ndim > 1 else dishonest_data[np.newaxis, :])

    # Stack data into single arrays
    honest_weights = np.vstack(honest_weights)
    dishonest_weights = np.vstack(dishonest_weights)

    # Combine all weights
    all_weights = np.concatenate([honest_weights, dishonest_weights])
    labels = np.concatenate([np.full(len(honest_weights), 1), np.full(len(dishonest_weights), 0)])

    all_weights = pca.transform(all_weights)
    y_pred = svm.predict(all_weights)
    score = f1_score(y_pred, labels)
    print(f'pre-trained SVM evaluation on DP updates with epsilon {epsilon} is {score}')


def read_shadow_updates(layer, n_components, degree, projection='PCA'):
    """
    Reads shadow model updates, performs PCA, and trains an SVM to classify honest and dishonest updates.

    Parameters:
        layer (int): The layer number for which updates are being read.
        n_components (int): The number of components for PCA.

    Returns:
        svm (SVC): The trained Support Vector Machine.
        pca (PCA): The trained PCA model.
    """
    honest_weights = []
    dishonest_weights = []

    # Filepaths
    honest_filename = f"delta_shadow_updates/GradAscent/Layer{str(layer)}/honest.csv"
    dishonest_filename = f"delta_shadow_updates/GradAscent/Layer{str(layer)}/dishonest.csv"
    visualize_high_dimensional_data(honest_filename, dishonest_filename)
    # Read data
    honest_data = np.loadtxt(honest_filename, delimiter=",")
    dishonest_data = np.loadtxt(dishonest_filename, delimiter=",")

    # Append data to respective lists
    honest_weights.append(honest_data if honest_data.ndim > 1 else honest_data[np.newaxis, :])
    dishonest_weights.append(dishonest_data if dishonest_data.ndim > 1 else dishonest_data[np.newaxis, :])

    # Stack data into single arrays
    honest_weights = np.vstack(honest_weights)
    dishonest_weights = np.vstack(dishonest_weights)

    # Combine all weights
    all_weights = np.concatenate([honest_weights, dishonest_weights])
    labels = np.concatenate([np.ones(len(honest_weights)), np.zeros(len(dishonest_weights))])
    if projection == 'PCA' :
        proj = PCA(n_components=n_components)
        proj.fit(all_weights)
        projected_weights = proj.transform(all_weights)

    if projection == 'sPCA' :
        proj, projected_weights = fhe_friendly_spca_mi(honest_weights, dishonest_weights, n_components=n_components)
        #projected_weights = proj.transform(all_weights)

    if projection == 'LDA' :
        proj = LDA(n_components=1)
        projected_weights = proj.fit_transform(all_weights, labels)
        projected_weights = proj.transform(all_weights)


    # Prepare labels for SVM training
    X_train, X_test, y_train, y_test = train_test_split(projected_weights, labels, test_size=0.25)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Train SVM
    if degree > 1 :
        svm = SVC(kernel="poly", degree=degree, gamma="auto", probability=False, C=5.0, verbose=False)  # Polynomial kernel
    else :
        svm = SVC(kernel="linear", class_weight="balanced", C=1.0, verbose=False)
    svm.fit(X_train, y_train)
    #print('\n\nsvm intercept_ ', svm.intercept_)
    y_pred = svm.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    num_support_vectors = svm.support_.shape[0]
    score = (100 * relu(f1 - 0.5)) ** 2
    score /= degree  * ((n_components * num_support_vectors)/4)
    evaluation = {
        "accuracy": np.round(f1, 5),
        "num_support_vectors": num_support_vectors,
        "svm_score ": np.round(score, 5)
    }

    # Return the trained models
    return svm, proj, evaluation, scaler


def fill_delta_datasets(x, y, n_models, use_dpsgd=False, noise_multiplier=1.0, l2_norm_clip=1.0, learning_rate=0.001):
    """
    Trains multiple models (honest and dishonest) and records their weight update differences (deltas).

    :param x: Input data
    :param y: Labels
    :param n_models: Number of models to train
    :param use_dpsgd: If True, uses differentially private SGD
    :param noise_multiplier: Noise level for DP-SGD
    :param l2_norm_clip: Gradient clipping norm for DP-SGD
    :param learning_rate: Learning rate for optimizer
    """
    batch_size=1
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}")

        # Create a new model instance
        model = MNIST_NN('lecun_uniform')  # Ensure this function is defined elsewhere

        # Select optimizer based on DP setting
        if use_dpsgd:
            optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=1,
                learning_rate=learning_rate
            )
            delta = 1/x.shape[0]
            epsilon, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(x.shape[0], batch_size, noise_multiplier, 1, delta)
            print(f"Estimated ε, delta: {epsilon, delta}")

        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        for epoch in range(12):
            print(f"Epoch {epoch+1}/12")

            # Clone the current model (before training) to track parameter updates
            original_model = clone_model(model)
            original_model.set_weights(model.get_weights())

            # Create dishonest model (for gradient ascent attack)
            dishonest_model = clone_model(model)
            dishonest_model.set_weights(model.get_weights())

            # Compile dishonest model with inverted learning rate
            dishonest_optimizer = tf.keras.optimizers.Adam(learning_rate=-learning_rate)
            dishonest_model.compile(optimizer=dishonest_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            # Train the honest model (normal update)
            model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0)

            # Train the dishonest model (gradient ascent)
            dishonest_model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0)

            # Compute model updates
            delta_honest_model = FedAvg_mnist([original_model, model], 2, [1, -1])  # Ensure FedAvg_mnist is defined
            delta_dishonest_model = FedAvg_mnist([original_model, dishonest_model], 2, [1, -1])

            # Save update differences (deltas) for key layers
            for layer_idx in [0, 2, 5, 6]:
                honest_weights = delta_honest_model.layers[layer_idx].get_weights()[0].flatten()
                dishonest_weights = delta_dishonest_model.layers[layer_idx].get_weights()[0].flatten()

                # Save honest updates
                with open(f"delta_shadow_updates/GradAscent/Layer{layer_idx}/honest.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(honest_weights)

                # Save dishonest updates
                with open(f"delta_shadow_updates/GradAscent/Layer{layer_idx}/dishonest.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(dishonest_weights)







def relu(x) :
    if x < 0 :
        return 0
    else :
        return x

def read_pca_from_csv(filepath='/home/akram/MetaClassifier/MetaClassifiers/Backdoor/PCA_matrix/pca_128.csv'):
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Locate sections
    components_start = lines.index("Components\n") + 1
    variance_start = lines.index("Explained Variance\n") + 1
    mean_start = lines.index("Mean\n") + 1

    # Extract components
    components = np.loadtxt(lines[components_start:variance_start - 1], delimiter=",")
    explained_variance = np.loadtxt(lines[variance_start:mean_start - 1], delimiter=",")
    mean = np.loadtxt(lines[mean_start:], delimiter=",")

    # Reconstruct PCA
    pca = PCA()
    pca.components_ = components
    pca.explained_variance_ = explained_variance
    pca.mean_ = mean
    return pca

def project_sample(pca, sample):
    # Center the sample using the mean
    centered_sample = sample - pca.mean_
    # Project onto principal components (matrix-vector multiplication)
    return np.dot(centered_sample, pca.components_.T)


def read_svm_from_csv(filepath='/home/akram/MetaClassifier/MetaClassifiers/GradientAscent/Model/svm_128.csv'):
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Parse CSV content
    support_vectors_start = lines.index("Support Vectors\n") + 1
    coefficients_start = lines.index("Coefficients\n") + 1
    biases_start = lines.index("Biases\n") + 1

    # Extract sections
    support_vectors = np.loadtxt(lines[support_vectors_start:coefficients_start - 1], delimiter=",")
    coefficients = np.loadtxt(lines[coefficients_start:biases_start - 1], delimiter=",")
    biases = np.loadtxt(lines[biases_start:], delimiter=",")

    # Reconstruct SVM
    svm = SVC(kernel='linear')  # or use the kernel you saved this SVM with
    svm.support_vectors_ = support_vectors
    svm.dual_coef_ = coefficients
    svm.intercept_ = biases
    return svm


import numpy as np

def sigmoid(x):

    return 1 / (1 + np.exp(-x))


def svm_infer_linear(svm, sample, t):
    # Extract support vectors, dual coefficients, and bias
    support_vectors = svm.support_vectors_
    coefficients = svm.dual_coef_.flatten()  # Flatten to ensure it's a 1D array
    b = svm.intercept_  # Bias term

    # Calculate the decision value by summing over the support vectors
    decision_value = 0
    for i in range(len(support_vectors)):
        decision_value += coefficients[i] * np.dot(support_vectors[i], sample)

    # Add the bias
    decision_value += b
    decision_value = -decision_value
    print("current decision value : ", decision_value)
    # Return the class based on the decision value
    return sigmoid(2 * decision_value)


def infer_from_model(model, svm, pca_matrix, t):
    """
    Perform inference using a DNN model, PCA matrix, and SVM.

    Steps:
    1. Extract the first layer weights from the model and flatten them.
    2. Project the flattened weights using the PCA matrix.
    3. Infer using the SVM on the PCA-projected weights.

    Parameters:
        model: A trained TensorFlow/Keras model.
        svm: A trained linear SVM (sklearn SVC with kernel='linear').
        pca_matrix: PCA transformation matrix (NumPy array).

    Returns:
        The predicted label from the SVM (0 or 1).
    """
    # Step 1: Extract and flatten the first layer weights
    first_layer_weights = model.layers[0].get_weights()[0]  # First layer weights
    flattened_weights = first_layer_weights.flatten()
    # Step 2: Project the flattened weights using the PCA matrix
    projected_weights = np.dot(flattened_weights, pca_matrix.T)

    # Step 3: Infer using the SVM
    predicted_label = svm_infer_linear(svm, projected_weights, t)

    return predicted_label


def finetuneSVM(svm, honest_models, dishonest_models, pca_matrix):
    """
    Fine-tune an SVM by retaining old support vectors and adding new data.
    """
    # Step 1: Prepare new data
    X_new = []
    y_new = []

    # Process honest models
    for model in honest_models:
        first_layer_weights = model.layers[0].get_weights()[0]
        flattened_weights = first_layer_weights.flatten()
        projected_weights = np.dot(flattened_weights, pca_matrix.T)
        X_new.append(projected_weights)
        y_new.append(1)

    # Process dishonest models
    for model in dishonest_models:
        first_layer_weights = model.layers[0].get_weights()[0]
        flattened_weights = first_layer_weights.flatten()
        projected_weights = np.dot(flattened_weights, pca_matrix.T)
        X_new.append(projected_weights)
        y_new.append(0)

    X_new = np.array(X_new)
    y_new = np.array(y_new)

    # Step 2: Retrieve old support vectors and labels
    old_support_vectors = svm.support_vectors_
    old_support_labels = svm.dual_coef_.flatten()

    # Map dual coefficients to binary labels (assuming a binary problem)
    old_labels = np.where(old_support_labels > 0, 1, 0)

    # Combine old and new data
    X_combined = np.vstack((old_support_vectors, X_new))
    y_combined = np.hstack((old_labels, y_new))

    # Step 3: Retrain the SVM
    svm.fit(X_combined, y_combined)

    return svm


def write_svm_to_csv(svm, filepath='/home/akram/MetaClassifier/MetaClassifiers/GradientAscent/Model/svm_128.csv') :
    support_vectors = svm.support_vectors_
    coefficients = svm.dual_coef_
    biases = svm.intercept_

    # Write parameters to CSV
    with open(filepath, "w") as f:
        f.write("Support Vectors\n")
        np.savetxt(f, support_vectors, delimiter=",")
        f.write("Coefficients\n")
        np.savetxt(f, coefficients, delimiter=",")
        f.write("Biases\n")
        np.savetxt(f, biases, delimiter=",")



# Function to load data from an npz file for a client
def load_client_data(npz_file):
    data = np.load(npz_file)
    x, y = data["x"], data["y"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)
    return (x_train, y_train), (x_test, y_test)

# Example usage for one client


def get_honest_data(n_clients) :
    training_data = []
    test_data = []

    for i in range(n_clients) :
        client_data_path = f"data/femnist/honest_data/client_{i}.npz"
        (x_train, y_train), (x_test, y_test) = load_client_data(client_data_path)

        x_train = x_train.astype('float32') / 1.0
        x_test = x_test.astype('float32') / 1.0

        y_train_cat = to_categorical(y_train, 10)
        y_test_cat = to_categorical(y_test, 10)

        training_data.append((x_train, y_train_cat))
        test_data.append((x_test, y_test_cat))

    return training_data, test_data

def get_honest_data_mnist(n_clients):
    """
    Splits the MNIST dataset among multiple clients and prepares the data for training.

    Parameters:
        n_clients (int): Number of clients to split the data among.

    Returns:
        training_data (list): List of tuples (x_train, y_train_cat) for each client.
        test_data (list): List of tuples (x_test, y_test_cat) for each client.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Split the training data among clients
    train_splits = np.array_split(np.arange(x_train.shape[0]), n_clients)
    test_splits = np.array_split(np.arange(x_test.shape[0]), n_clients)

    training_data = []
    test_data = []

    for i in range(n_clients):
        x_train_client = x_train[train_splits[i]]
        y_train_client = y_train_cat[train_splits[i]]

        x_test_client = x_test[test_splits[i]]
        y_test_client = y_test_cat[test_splits[i]]

        training_data.append((x_train_client, y_train_client))
        test_data.append((x_test_client, y_test_client))

    return training_data, test_data

def get_dishonest_data(n_cliets) :

    training_data = []
    test_data = []
    fracs = [0.9, 0.8, 0.7, 0.6, 0.5]
    for frac in fracs :
        filename = 'data/byzantine/backdoored_mnist_frac='+str(frac)+'.npz'
        data = np.load(filename)

        # Extract the training and testing sets
        x_train_backdoor = data['x_train']
        y_train_backdoor = to_categorical(data['y_train'], 10)
        x_test_backdoor = data['x_test']
        y_test_backdoor = to_categorical(data['y_test'], 10)
        training_data.append((x_train_backdoor, y_train_backdoor))
        test_data.append((x_test_backdoor, y_test_backdoor))

    return training_data, test_data


def grid_search() :
    for filtering_layer in [0, 2, 5, 6] :
        for dim in [8, 64, 128] :
            for deg in [1, 2, 3, 4] :
                svm, pca, eval, _ = read_shadow_updates(filtering_layer, dim, deg, projection='sPCA')
                print(f'svm layer {filtering_layer} dim : {dim} deg : {deg} eval : {eval}')



def run_training(honest_data, dishonest_data, local_epochs, rounds, seed, use_svm, lr_decay=True, use_dpsgd=False) :

    scores = []
    scores_backdoor = []

    if use_svm :
        grid_search()
        filtering_layer = 0
        svm, pca, eval, scaler = read_shadow_updates(filtering_layer, 8, 1, projection='PCA')
        print(f'svm to used has eval : {eval}')
        test_on_dp_updates(svm, pca, filtering_layer, 0.491)
        test_on_dp_updates(svm, pca, filtering_layer, 0.31)
        file_name1 = f'convergence_data_with_dp/GradAscent/with_svm_filter/honest_scores_seed_{seed}.csv'
    elif not use_svm :
        file_name1 = f'convergence_data_with_dp/GradAscent/without_svm_filter/honest_scores_seed_{seed}.csv'

    honest_training_data, honest_test_data = honest_data

    n_honest_clients = len(honest_training_data)
    n_dishonest_clients = 0

    x_test_honest = np.concatenate([data[0] for data in honest_test_data], axis=0)
    y_test_honest = np.concatenate([data[1] for data in honest_test_data], axis=0)
    if dishonest_data != [] :
        dishonest_training_data, dishonest_test_data = dishonest_data
        n_dishonest_clients = len(dishonest_training_data)
        x_test_dishonest = np.concatenate([data[0] for data in dishonest_test_data], axis=0)
        y_test_dishonest = np.concatenate([data[1] for data in dishonest_test_data], axis=0)
    n_clients = n_honest_clients + n_dishonest_clients
    batch_size=12
    global_model = MNIST_NN('lecun_uniform')

    with open(file_name1, mode="a", newline='') as f1 :
        writer1 = csv.writer(f1)
        #Main training loop
        lr=0.12
        for t in range(rounds):
            print(f'\nRound {t} ...')
            honest_models = []
            dishonest_models = []

            if lr_decay :  #Local learning rate for this round
                lr=0.05/(2*(t+1)) #Start strong and decay quick
            else :
                lr*=(0.95)
                print(f'\n\ncurrent learning rate {lr}')
            #Update honest clients
            delta_honest_models = []
            for i in range(n_honest_clients):
                model = update_local_model(global_model, (28, 28, 1), lr=lr, use_dpsgd=use_dpsgd)

                start_time = time.time()  # Start timer
                model.fit(honest_training_data[i][0], honest_training_data[i][1], epochs=local_epochs, batch_size=batch_size, verbose=0)
                end_time = time.time()  # End timer
                print(f"Local update time: {end_time - start_time:.6f} seconds ds_size {honest_training_data[i][0].shape[0]}")
                #fill_delta_datasets(honest_training_data[i][0], honest_training_data[i][1], 1, use_dpsgd=False)
                print(f'Updating client {i} ... ', model.evaluate(honest_test_data[i][0], honest_test_data[i][1], verbose=0))
                honest_models.append(model)
                delta_model = FedAvg_mnist([global_model, model], 2, [1, -1])
                delta_honest_models.append(delta_model)
                """
                for i in [0, 2, 5, 6] :
                    first_layer_weights = delta_model.layers[i].get_weights()[0]  # First layer weights
                    flattened_weights = first_layer_weights.flatten()

                    with open(f"delta_shadow_updates/GradAscent/Layer{i}/honest.csv", "a", newline="") as f:
                        writer = csv.writer(f)

                        writer.writerow(flattened_weights)
                """
            #Update dishonest clients (if any)
            delta_dishonest_models = []
            for i in range(n_dishonest_clients):
                model = update_local_model(global_model, (28, 28, 1), lr=-lr, use_dpsgd=use_dpsgd)
                model.fit(dishonest_training_data[i][0], dishonest_training_data[i][1], epochs=local_epochs, batch_size=batch_size, verbose=0)
                print(f'Updating client {i} ... ', model.evaluate(dishonest_test_data[i][0], dishonest_test_data[i][1], verbose=0))
                dishonest_models.append(model)
                delta_model = FedAvg_mnist([global_model, model], 2, [1, -1])
                delta_dishonest_models.append(delta_model)
                """
                for i in [0, 2, 5, 6] :
                    first_layer_weights = delta_model.layers[i].get_weights()[0]  # First layer weights
                    flattened_weights = first_layer_weights.flatten()

                    with open(f"delta_shadow_updates/GradAscent/Layer{i}/dishonest.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(flattened_weights)
                """
            #Filter (if use_svm = True) then aggregate
            models = honest_models + dishonest_models
            delta_models = delta_honest_models + delta_dishonest_models
            if use_svm:
                property_filters = []
                print('current decision values : ')
                for m in delta_models :
                    property_filters.append(svm.decision_function(10 * pca.transform(m.layers[filtering_layer].get_weights()[0].flatten().reshape(1, -1))))
                    print(property_filters[-1])

                current_avg = sum(property_filters)/n_clients
                print("current average : ", current_avg)
                property_filters = [sigmoid(10 * (x-0)) for x in property_filters]
                #save filter values
                with open('convergence_data_with_dp/GradAscent/honest_filters.csv' , "a") as filter_f1, open('convergence_data_with_dp/GradAscent/dishonest_filters.csv', "a") as filter_f2 :
                    honest_filter_writer = csv.writer(filter_f1)
                    dishonest_filter_writer = csv.writer(filter_f2)
                    honest_filter_writer.writerows([property_filters[0:n_honest_clients]])
                    dishonest_filter_writer.writerows([property_filters[n_honest_clients:n_clients]])

                filtered_fedavg_weights = [(property_filters[i] * np.round(1 / n_clients, 2)) for i in range(n_clients)]
                fedavg_weights = [i / sum(filtered_fedavg_weights) for i in filtered_fedavg_weights]
                print('Normalized filtered weights: ', fedavg_weights)
                global_model = FedAvg_mnist(models, n_clients, fedavg_weights)
            else:
                global_model = FedAvg_mnist(models, n_clients, [np.round(1 / n_clients, 5) for i in range(n_clients)])
            # Append scores for honest and dishonest data

            print('model 0 evaluation ', models[0].evaluate(x_test_honest, y_test_honest, verbose=0))
            honest_score = global_model.evaluate(x_test_honest, y_test_honest, verbose=0)
            scores.append([honest_score])  # Wrap in a list for writing rows
            print("Honest: ", honest_score)

            if dishonest_data:
                backdoor_score = global_model.evaluate(x_test_dishonest, y_test_dishonest, verbose=0)
                scores_backdoor.append(backdoor_score)  # Wrap in a list for writing rows
                print("Byzantine : ", backdoor_score)
                #writer2.writerows([backdoor_score])
            writer1.writerows([honest_score])



def display_image(image):
    plt.imshow(image, cmap="gray")  # Display the image in grayscale
    plt.axis("off")  # Turn off axes for better visualization
    plt.show()


if __name__ =='__main__' :

    #Femnist datasets
    honest_training_data, honest_test_data =  get_honest_data(2)
    honest_data = (honest_training_data, honest_test_data)

    dishonest_training_data, dishonest_test_data =  get_honest_data_mnist(2)
    #datasets containing backdoors
    dishonest_data = (dishonest_training_data, dishonest_test_data)

    max_seeds = 5
    #run_training(honest_data, [], local_epochs=1, rounds=30, seed=0, use_svm=False, lr_decay=True, use_dpsgd=True)
    for seed in range(1, max_seeds) :
        run_training(honest_data, dishonest_data, local_epochs=1, rounds=30, seed=seed, use_svm=True, lr_decay=False, use_dpsgd=True)
        #run_training(honest_data, dishonest_data, local_epochs=1, rounds=30, seed=seed, use_svm=True, lr_decay=False, use_dpsgd=True)
