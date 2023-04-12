
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

# Euros = np.array([1,3,5,7,10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
# Dolares = np.array([1.8,3.23,5.59,7.55,10.78, 21.56, 36.48, 48.64, 60.80, 72.96, 85.12, 97.28], dtype=float)

# # Definir capas y modelo por conversión
# capa_EuDo1 = tf.keras.layers.Dense(units=3, input_shape=[1])
# capa_EuDo2 = tf.keras.layers.Dense(units=3)
# capa_EuDo_salida = tf.keras.layers.Dense(units=1)
# modelo_EuDo = tf.keras.Sequential([capa_EuDo1, capa_EuDo2, capa_EuDo_salida])

# # Compilar modelo
# modelo_EuDo.compile(
#     optimizer=tf.keras.optimizers.Adam(0.1),
#     loss='mean_squared_error'
# )

# # Entrenamiento
# historial_EuDo = modelo_EuDo.fit(Euros, Dolares, epochs=400, verbose=False)
# tfjs.converters.save_keras_model(modelo_EuDo,'modelo_EuDo')

# print( modelo_EuDo.predict([[1]]))





# # Definir valores de entrada
# Litros = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
# Galones = np.array([2.64, 5.28, 7.92, 10.56, 13.20, 15.84, 18.48, 21.12], dtype=float)

# # Definir capas y modelo por conversión
# capa_LiGa1 = tf.keras.layers.Dense(units=3, input_shape=[1])
# capa_LiGa2 = tf.keras.layers.Dense(units=3)
# capa_LiGa_salida = tf.keras.layers.Dense(units=1)
# modelo_LiGa = tf.keras.Sequential([capa_LiGa1, capa_LiGa2, capa_LiGa_salida])

# # Compilar modelo
# modelo_LiGa.compile(
#     optimizer=tf.keras.optimizers.Adam(0.1),
#     loss='mean_squared_error'
# )

# # Entrenamiento
# historial_LiGa = modelo_LiGa.fit(Litros, Galones, epochs=400, verbose=False)
# tfjs.converters.save_keras_model(modelo_LiGa,'modelo_LiGa')

# print( modelo_LiGa.predict([[10]]))




# Definir valores de entrada
mA = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=float)
A = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)

# Definir capas y modelo para conversión de mA a A
capa_mA_A2 = tf.keras.layers.Dense(units=4, input_shape=[1])
capa_mA_A1 = tf.keras.layers.Dense(units=4)
capa_mA_A_salida = tf.keras.layers.Dense(units=1)
modelo_mA_A = tf.keras.Sequential([capa_mA_A2,capa_mA_A1 ,capa_mA_A_salida])

# Compilar modelo
modelo_mA_A.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenamiento
historial_mA_A = modelo_mA_A.fit(A, mA , epochs=500, verbose=False)
tfjs.converters.save_keras_model(modelo_mA_A, 'modelo_mA_A')

print(modelo_mA_A.predict([[100]]))
