import os
os.system("cls" if os.name in ("nt", "dos") else "clear")

import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

gestos=["A","An","C","Co","I","M"]
y=0;Y=[];fila=[]
for g in gestos:
    ne=0
    for n in range(5):
        senial=g+str(n+1)+"_EMG.txt"
        with open('GitHub/'+senial, 'r', encoding='utf-8') as file:
            archivo = file.read()
            for f in archivo.split("\n"):
                columna=[]
                for c in f.split("  "):
                    if c!="":
                        columna.append(float(c))
                umbra=sum(abs(ac) for ac in columna)/8
                if umbra>=0.03:
                    columna.append(umbra)
                    fila.append(columna)
                    ne+=1
                    Y.append(y)
    y+=1
    print(ne)

X=np.array(fila)
Y=np.array(Y)

# Crea el modelo de red neuronal con una capa oculta
mlp = MLPClassifier(hidden_layer_sizes=((100,30,10)), 
                    activation='logistic', 
                    solver='sgd', 
                    learning_rate_init=0.3,
                    max_iter=90000)

# Entrena el modelo con TODOS los datos
# mlp.fit(X, Y)

# Entrena el modelo en lotes y guarda los valores de pérdida
loss_values = []
classes = np.unique(Y)
print(classes)

mlp.partial_fit(X[:1], Y[:1], classes=classes)
loss_values.append(mlp.loss_)
for i in range(1, mlp.max_iter):
    mlp.partial_fit(X, Y)
    loss_values.append(mlp.loss_)

# Evalúa el modelo (no se necesita un conjunto de prueba, pero puedes usar el mismo conjunto para evaluar)
accuracy = mlp.score(X,Y)
print("Precisión:", accuracy)

# Evalúa el modelo con entradas arbitrarias
prb1 = X[30:60,:]; sol1=Y[30:60]
prb2 = X[1242+30:1242+60,:];sol2=Y[1242+30:1242+60]
prb3 = X[2555+30:2555+60,:];sol3=Y[2555+30:2555+60]
prb4 = X[4182+30:4182+60,:];sol4=Y[4182+30:4182+60]
prb5 = X[5516+30:5516+60,:];sol5=Y[5516+30:5516+60]
prb6 = X[6073+30:6073+60,:];sol6=Y[6073+30:6073+60]

predicciones = mlp.predict(np.concatenate((prb1,prb2,prb3,prb4,prb5,prb6),axis=0))
soluciones = np.concatenate((sol1,sol2,sol3,sol4,sol5,sol6))
cps=0
for ps in range(len(soluciones)):
    print(predicciones[ps]," --- ",soluciones[ps])
    if predicciones[ps]==soluciones[ps]:
        cps+=1
print("porsentaje: ",cps/len(soluciones))

# Grafica la curva de pérdida
plt.plot(loss_values)
plt.xlabel("Iteraciones")
plt.ylabel("Pérdida")
plt.title("Curva de Pérdida durante el Entrenamiento XOR")
plt.show()


# Guardar el modelo
filename = 'mlp_model.pkl'
pickle.dump(mlp, open(filename, 'wb'))  # 'wb' es para escritura binaria

