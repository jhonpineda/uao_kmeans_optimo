import random
import os
import numpy as np
import cv2 as cv
#from google.colab.patches import cv2_imshow
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

def shoes_kaggle():
    etiquetas = os.listdir('dataset/train')
    #print(etiquetas)

    train_images = []
    train_labels = []

    for categoria in etiquetas:
        print("Leyendo categoria", categoria)
        class_num = etiquetas.index(categoria)
        ruta = os.path.join('dataset/train', categoria)
        for img in tqdm(os.listdir(ruta)):
            img_array = cv.imread(os.path.join(ruta, img), cv.IMREAD_GRAYSCALE)
            # los descriptores de la clase funcionan mejor en escala de grises con los mismos resultados
            new_array = cv.resize(img_array, (400, 400))
            # cv.resize(400, 400) Para que quede en dos dimensiones
            train_images.append(new_array)
            train_labels.append(class_num)

    x_train = np.array(train_images)
    y_train = np.array(train_labels)

    imagenes = {}
    folder = 'dataset/train'
    for filename in os.listdir(folder):
        categoria = []
        ruta = folder + "/" + filename
        for cat in os.listdir(ruta):
            img = cv.imread(ruta + "/" + cat, cv.IMREAD_GRAYSCALE)
            if img is not None:
                categoria.append(img)
        imagenes[filename] = categoria

    train = imagenes

    sift_vectores = {}  # Tiene el descriptor y la categoria
    descriptor_list = []
    sift = cv.SIFT_create()

    for key, value in imagenes.items():
        features = []
        for img in value:
            kp, desc = sift.detectAndCompute(img, None)
            descriptor_list.extend(desc)  # descriptores sin organizacion
            features.append(desc)
        sift_vectores[key] = features

    descriptor_list = descriptor_list
    train_bovw_feat = sift_vectores

    suma_distancias_cuadradas = []
    K = range(5, 200, 1)
    for k in K:
        #print(k)
        km = KMeans(n_clusters=k)
        km.fit(descriptor_list)
        suma_distancias_cuadradas.append(km.inertia_)

    plt.plot(K, suma_distancias_cuadradas, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    #plt.show()

    plt.savefig('result_k.png')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    shoes_kaggle()


