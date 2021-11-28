import csv
import scipy.io
import cv2
import numpy as np
from os import listdir
import os.path

def create_blank(width, height, rgb_color=(255, 255, 255)):

    image = np.zeros((height, width, 3), np.uint8)

    color = tuple(reversed(rgb_color))
    
    image[:] = color

    return image

def promedio_Vector(vector):
    
    vOriginal = vector
    
    index_min = np.argmin(vector)
    index_max = np.argmax(vector)
    
    if index_max != 0:
        index_max = index_max - 1
    
    vOriginal = np.delete(vOriginal, index_min, 0)
    vOriginal = np.delete(vOriginal, index_max, 0)
    
    kind = np.around(np.mean(np.array(vOriginal).astype(np.float)))
    
    if kind >=0 and kind <=10:
        kind = 1
    
#     if kind >=8 and kind <=10:
#         kind = 2
        
    if kind >=11 and kind <=15:
        kind = 2
    
    return kind

def showResult(kind):
    reader = csv.reader(open('class.csv', 'r'))
    
    for index,row in enumerate(reader):
        if int(row[0]) == kind:
            return row[1]
        
def deleteDir(directorio):
    for archivo in listdir(directorio):
        if os.path.exists(directorio + '/' + archivo):
            os.remove(directorio + '/' + archivo)
    os.rmdir(directorio)
            

def principal(directorio):
    
    frame_w = 641
    frame_h = 481
    color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    arreglo = []
    
    resultado = create_blank(frame_w, frame_h, rgb_color=color)
    
    directorio1 = directorio + '/Scores/'
    for archivo in listdir(directorio1):
        reader = csv.reader(open(directorio1 +'/'+ archivo, 'r'))
        for index,row in enumerate(reader):
            if row[0] == "Aciertos":
                arreglo.append(row[1])
               

    cv2.putText(resultado, "Resultado:", (20, 60), font, 0.6, 244, 2, 4)
    cv2.putText(resultado, "--------------------------", (20, 100), font, 0.6, 244, 2, 4)
    cv2.putText(resultado, "En este ejercicio tienes un " + showResult(promedio_Vector(arreglo)) + ".", (20, 120), font, 0.6, 244, 2, 4)
    cv2.putText(resultado, "Presione cualquier tecla para salir...", (20, 160), font, 0.6, 244, 2, 4)
    cv2.imshow("Laparoscopia: Ejercicio 2", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()