import inline as inline
import matplotlib
#%matplotlib inline

#Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

x = 2*np.random.rand(100,1)
y = 4+3*x+np.random.randn(100,1)
print("La longitud del conjunto es: ", len(x))

#Representación gráfica .
#plt.plot(x,y,"b.")
#plt.show()
plt.plot(x,y,"b.")
plt.xlabel("Equipos afectados (u/1000)")
plt.ylabel("Coste del incidente (u/10000)")
plt.show()

#Diccionario (los 10 primeros)
data = {'n_equipos_afectados': x.flatten(), 'coste': y.flatten()}
df = pd.DataFrame(data)
print(df.head(10))

#Escalado del número de equipos afectados
df['n_equipos_afectados'] = df['n_equipos_afectados']*1000
df['n_equipos_afectados'] = df['n_equipos_afectados'].astype('int')

#Escalado del coste
df['coste'] = df['coste']*10000
df['coste'] = df['coste'].astype('int')
print(df.head(10))

#Representación gráfica del conjunto de datos
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()

#Construccioón del modelo y ajuste de la función hipotesis
lin_reg = LinearRegression()
lin_reg.fit(df['n_equipos_afectados'].values.reshape(-1, 1), df['coste'].values)

#Parámetro theta 0
print(lin_reg.intercept_)

#Parámetro theta 1
print(lin_reg.coef_)

#Predicción para el valor mínimo y máximo del conjunto de datos de entrenamiento
x_min_max = np.array([[df["n_equipos_afectados"].min()], [df["n_equipos_afectados"].max()]])
y_train_pred = lin_reg.predict(x_min_max)

#Representación gráfica de la función hipotesis generada
plt.plot(x_min_max, y_train_pred, "g-")
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()

#Nuevos ejemplos
x_new = np.array([[1600]]) #1600 equipos afectados
coste = lin_reg.predict(x_new) #prediccion del coste que tendria el incidente
print("El coste del incidente seria: ", int(coste[0]), "€")

#Representacion gráfica del nuevo ejemplo
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.plot(x_min_max, y_train_pred, "g-")
plt.plot(x_new, coste, "rx")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()
