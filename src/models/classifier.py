from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def Classifier(real_train_labels, real_train_images, generated_labels, generated_samples, k):

    # ------------
    # Form training set from real images
    # ------------

    #k = 5
    knn = KNeighborsClassifier(n_neighbors = k)

    #knn.fit (real_train_images, real_train_labels)

    knn.fit (generated_samples, generated_labels)

    predicted_labels = knn.predict(real_train_images)

    accuracy = np.mean(predicted_labels == real_train_labels)

    return accuracy

    #print(f'Classification accuracy: {accuracy}')