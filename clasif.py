import csv
import scipy.io
import cv2
import numpy as np
from os import listdir
import os.path
from scipy import stats as st
import joblib 


def create_blank(width, height, rgb_color=(255, 255, 255)):

    image = np.zeros((height, width, 3), np.uint8)

    color = tuple(reversed(rgb_color))
    
    image[:] = color

    return image

def promedio_Vector(vector):
    
    vOriginal = vector    
    #Calcular la moda de clasificación
    kind = np.bincount(vOriginal).argmax()
    kind1=st.mode(vOriginal)[0]
    #print(kind, int(kind1))
   
    if kind == 1:
        kind = "Nivel experto"
    elif kind==2:
        kind = "Nivel inexperto"

    return kind
            

def principal(directorio):
    
    directorio1 = directorio + '/Maps'
  
    frame_w = 641
    frame_h = 481
    color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    arreglo = []
    
    resultado = create_blank(frame_w, frame_h, rgb_color=color)
    
    
         # Cargar el modelo.
    knn1 = joblib.load('modelo_entrenado.pkl')
        #Lectura de archivos
    for root,dirs,files in os.walk(directorio1):
        for file in files:
            
            if file.endswith(".csv") and file!='clasif.csv':
                #Cargar datos
                x = np.genfromtxt(directorio1 +"/" +file,delimiter=',')
                #print(np.where(np.isnan(x)))
                x[np.isnan(x)]=0
                X0 = np.array(x.reshape(-1, 14))
                #Realizar predicción de la clase a la que pertenece
                pred1 = knn1.predict(X0)[0]
                f2 = open(directorio1+"/clasif.csv", "a")
                f2.write("Clasif" + "," + str(int(pred1)) + "\n")
                #f2.write("Clasif" + "," + str(int(pred2)) + "\n")
                f2.close()
    
    
    
    reader = csv.reader(open(directorio1 +'/'+ 'clasif.csv', 'r'))
    for index,row in enumerate(reader):
        if row[0] == "Clasif":
            arreglo.append(row[1])
               

    cv2.putText(resultado, "Resultado:", (20, 60), font, 0.6, 244, 2, 4)
    cv2.putText(resultado, "--------------------------", (20, 100), font, 0.6, 244, 2, 4)
    cv2.putText(resultado, "En este ejercicio tienes un " + promedio_Vector(arreglo) + ".", (20, 120), font, 0.6, 244, 2, 4)
    cv2.putText(resultado, "Presione cualquier tecla para salir...", (20, 160), font, 0.6, 244, 2, 4)
    cv2.imshow("Laparoscopia: Ejercicio 3", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()