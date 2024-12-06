import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

def main():
    # Cargar el conjunto de datos Iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Convertir a DataFrame para análisis
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = y

    # Análisis de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.show()

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar las características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Crear el modelo K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(n_neighbors=3)

    # Entrenar el modelo
    knn.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = knn.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy * 100:.2f}%')

    # Mostrar el informe de clasificación
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Predecir la clase de una nueva muestra
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    new_sample_scaled = scaler.transform(new_sample)
    prediction = knn.predict(new_sample_scaled)
    print(f'\nPredicción para la nueva muestra: {iris.target_names[prediction][0]}')

    # Análisis de características con RFE y SVM
    svc = SVC(kernel="linear")
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(X, y)

    print("\nRanking de características según RFE:")
    for i in range(len(feature_names)):
        print(f'{feature_names[i]}: {rfe.ranking_[i]}')

if __name__ == "__main__":
    main()
