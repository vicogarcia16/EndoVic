#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time as tiempo
from time import time
import os.path
import sys
import csv
import decimal
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from math import *
from scipy import signal
import clasif
from matriz_conversion import conversion_1p

# Varoables globales
global iniciar, posiciones1
posiciones1 = []
iniciar = 0

# Estilo de letra
font = cv2.FONT_HERSHEY_SIMPLEX

# Ancho y alto del video
frame_w = 640
frame_h = 480

# Configuración de la ventana Tkinter

def center(ventana):
    ventana.update_idletasks()
    w = ventana.winfo_width()
    h = ventana.winfo_height()
    extraW = ventana.winfo_screenwidth()-w
    extraH = ventana.winfo_screenheight()-h
    ventana.geometry("%dx%d%+d%+d" % (w, h, extraW/2, extraH/2))

# Creación del directorio de alojamiento de datos

def createDir(directorio):
    if os.path.exists(directorio):
        messagebox.showwarning(
            "Advertencia", "El nombre de usuario ya existe.")
    else:
        if e1.get() != "":
            os.mkdir(directorio)
            os.mkdir(directorio +'/Repeticiones')
            os.mkdir(directorio +'/Maps')
            os.mkdir(directorio +'/Graficas')
            coordenadas = dibujar()
            camara(coordenadas)
        else:
            messagebox.showwarning(
                "Advertencia", "No ha indicado un nombre de usuario.")

def abrir():
    if e1.get() != "":
        directory = e1.get() + '-Ejercicio-3-1'
       
        clasif.principal(directory)
       

    else:
        messagebox.showwarning(
                "Advertencia", "No ha indicado un nombre de usuario.")

def normalizar(ruta): 
    coordenadas = []
    nueva = []
    ruta_nueva = ruta +'-nr'
    reader = csv.reader(open(ruta+".csv", "r"))
    for index,row in enumerate(reader):
        if row[0] != 'TF':
            coordenadas.append((int(row[0]),int(row[1]),int(row[2])))
            
    for i in coordenadas:
        if i not in nueva:
            fo= open(ruta_nueva + ".csv", "a")
            fo.write(str(i[0])+","+ str(i[1])+ ","+str(i[2]) + "\n")
            nueva.append((i[0],i[1],i[2]))
            
    reader = csv.reader(open(ruta +".csv", "r"))
    for index,row in enumerate(reader):
        if row[0] == 'TF':
            fo= open(ruta_nueva + ".csv", "a")
            fo.write(row[0]+","+ row[1]+","+row[2]+ "\n")

    return ruta_nueva

def grafica(ruta, ruta2, prueba):
    
    X = np.genfromtxt(ruta + '.csv', delimiter=',')
    x = X[0:-1:,0]
    y = X[0:-1:,1]
    z = X[0:-1:,2]

    fig = plt.figure(figsize=(8,6))
    # Creamos el plano 3D
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.plot(x,z,y, linestyle='-', color = 'black', linewidth = 1.5)
    ax1.view_init(-165,-65)
    #plt.subplots_adjust(left = 0.05)
    #ax1.plot_wireframe(x2, y2, y22)
    plt.title('Gráfica de movimiento del usuario - Repetición '+str(prueba))
    ax1.set_xlabel('eje x')
    ax1.set_ylabel('eje z')
    ax1.set_zlabel('eje y')
    plt.savefig(ruta2 + '.png')
    #plt.show()
    

def maps(ruta, ruta1):
    X = np.genfromtxt(ruta + '.csv', delimiter=',')
    x = X[0:-1:,0]
    y = X[0:-1:,1]
    z = X[0:-1:,2]
    t = X[-1:,1]
    t1 = t[-1:1]
    t1 = str(t).strip('[]')
    
    #Cálculo de MAP's"
    for i in range(len(x)):
        x[i]=(x[i])/100
        y[i]=(y[i])/100
        z[i]=(z[i])/100
        
    #EndoViS Path Length
    #Derecha
    dist = np.sqrt(np.diff(x,1)**2 + np.diff(y,1)**2 + np.diff(z,1)**2)
    PLD=np.sum(dist)

    #EndoViS Depth Perception
    #Derecha
    dist = np.sqrt(np.diff(y,1)**2 + np.diff(z,1)**2)
    DPD=np.sum(dist)

    #EndoViS Motion Smoothness
    #Derecha
    MS = np.sum((np.diff(x,3)**2 + np.diff(y,3)**2 + np.diff(z,3)**2))
    MS1=np.sqrt(0.5*(MS))
    #Carvalo y EVA
    cte = (t**5)/(2*PLD**2)
    MS2 = np.sqrt(cte*(MS))

    # Resampleo de la señal a cada segundo
    num = round(len(x)/30)
    f = round(len(x)/num)
    xx = signal.resample_poly(x,1,f,window = ('kaiser',3.2))
    yy = signal.resample_poly(y,1,f,window = ('kaiser',2.6))
    zz = signal.resample_poly(z,1,f,window = ('kaiser',0.5))
    #xx = signal.resample(x, num)
    #yy = signal.resample(y, num)
    #zz = signal.resample(z, num)

    #Se convierten los datos en centimetros *0.042
    #   (si se lee con EndoMIIDT) y posterior a milimetros
    xx = xx*1000
    yy = yy*1000
    zz = zz*1000


    #EndoViS Average Speed (mm/s)
    #Derecha
    SpeedD = np.sqrt(np.diff(xx,1)**2 + np.diff(yy,1)**2 + np.diff(zz,1)**2)
    Mean_SpeedD = np.mean(SpeedD)
    #print("\nEndoViS Average Speed (mm/s): ", Mean_SpeedD)


    #EndoViS Average Acceleration (mm/s^2)
    #Derecha
    Accd = np.sqrt(np.diff(xx,2)**2 + np.diff(yy,2)**2 + np.diff(zz,2)**2)
    Mean_AccD = np.mean(Accd)
    #print("\nEndoViS Average Acceleration (mm/s^2): ", Mean_AccD)


    #EndoViS Idle Time (%)
    #Derecha
    idle1D = np.argwhere(SpeedD<=5)
    idleD =(len(idle1D)/len(SpeedD))*100
    #print("\nEndoViS Idle Time (%): ", idleD)



    #EndoViS Max. Area (m^2)
    #Derecha
    max_horD = max(x)-min(x)
    max_vertD = max(y)-min(y)
    MaxAreaD = max_vertD*max_horD

    #EndoViS Max. Volume (m^3)
    #Derecha
    max_altD = max(z)-min(z)
    MaxVolD = MaxAreaD*max_altD

    #EndoViS Area/PL : EOA
    #Derecha
    A_PLD = np.sqrt(MaxAreaD)/PLD
    #print("\nEndoViS Economy of Area (au.): ", A_PLD)

    #EndoViS Volume/PL: EOV
    #Derecha
    A_VD =  MaxVolD**(1/3)/PLD
    #print("\nEndoViS Economy of Volume (au.): ", A_VD)


    #EndoViS Energia
    #Derecha
    EXv = np.sum(x**2)
    EYv = np.sum(y**2)
    EZv = np.sum(z**2)

    EndoEAD = (EXv+EYv)/(MaxAreaD*100) #J/cm^2
    EndoEVD = (EXv+EYv+EZv)/(MaxVolD*100) #J/cm^3
    #print("\nEndoViS Energy of Area (J/cm^2.): ", EndoEAD)
    #print("\nEndoViS Energy of Volume (J/cm^3.): ", EndoEVD)


    # Print parameters
    #EndoViS Tiempo
    print("\nEndoViS Tiempo (s): ", t)
    print("EndoViS Path Length (m.): ", PLD)
    print("EndoViS Depth Perception (m.): ", DPD)
    print("EndoViS Depth Perception along trocar", None)
    print('EndoViS Motion Smoothness 1: ', MS1)
    print('EndoViS Motion Smoothness 2: ', MS2)
    print("EndoViS Average Speed (mm/s): ", Mean_SpeedD)
    print("EndoViS Average Acceleration (mm/s^2): ", Mean_AccD)
    print("EndoViS Idle Time (%): ", idleD)
    print("EndoViS Economy of Area (au.): ", A_PLD)
    print("EndoViS Economy of Volume (au.): ", A_VD)
    print("EndoViS Bimanual Dexterity", None)
    print("EndoViS Energy of Area (J/cm^2.): ", EndoEAD)
    print("EndoViS Energy of Volume (J/cm^3.): ", EndoEVD)


    fo = open(ruta1 + ".csv", "a")
    fo.write(t1 + "," + str(PLD) + "," + str(DPD) + "," +  "0"  + "," + str(MS1) + "," + str(MS2).strip('[]') + "," + str(Mean_SpeedD) + "," + str(Mean_AccD) + "," + str(idleD) + "," + str(A_PLD) + "," + str(A_VD) + "," + "0"  + "," + str(EndoEAD) + "," + str(EndoEVD) + "\n")
    fo.close

# Función de inicio del proceso con la cámara
def dibujar():
    lista = []
    cap = cv2.VideoCapture(4,cv2.CAP_V4L)
    _, imagen = cap.read()
    
    # Aplicar desenfoque para eliminar ruido
    frame = cv2.blur(imagen, (7, 7))
    # Convertimos la imagen a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    azul_bajos = np.array([100, 100, 25])
    azul_altos = np.array([120, 255, 255])
    fondo = cv2.inRange(hsv, azul_bajos, azul_altos)

    # Invertimos la mascara para obtener las bolas
    bolas = fondo
   
    # Eliminamos ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 , 4))
    eros = cv2.erode(bolas, kernel, iterations=2)
    maskOpen = cv2.morphologyEx(eros, cv2.MORPH_OPEN, kernel)
    dila = cv2.dilate(maskOpen,None,iterations=2)
    bolas1 = cv2.morphologyEx(dila, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bolas1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    for cnt in contours:
        max_area =  0
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            cnt_best = cnt
            #area1 = moments['m00']
            #área de la pantalla
            #print(area)
            
        if(max_area > 1000 and max_area< 2500):
            contours_rojos = cnt_best
            
            #Calcular el centro a partir de los momentos
            momentos = cv2.moments(cnt_best)
            cx = int(momentos["m10"] / momentos["m00"])
            cy = int(momentos["m01"] / momentos["m00"])
            #cv2.drawContours(imagen, contours, -1, (255,255,0), 3)
            total+=1
            lista.append((cx,cy))

            print('Área {}: {}'.format(total, max_area))
            print('Centroide lista: {}'.format(lista[-1]))
            print('Centroide tupla: {}'.format((cx,cy)))

        #cv2.drawContours(imagen, contours, -1, (0, 255, 0), 2)
                #cv2.drawContours(imagen, contours_rojos, -1, (0, 0, 255), 2)
    cap.release()
    return lista
        
def camara(coordenadas):
    cx = 0
    cy = 0

    cx1 = 0
    cy1 = 0

    lastX = -1
    lastY = -1

    lastX1 = -1
    lastY1 = -1
    
    cx2 = 0
    cy2 = 0

    cx3 = 0
    cy3 = 0

    lastX2 = -1
    lastY2 = -1

    lastX3 = -1
    lastY3 = -1


    if iniciar != 1:
        # Propiedades del video
        FRAME_PROP_WIDTH = 3
        FRAME_PROP_HEIGHT = 4

        # Variables
        inicio = 0
        fin = 0
        detener = 0
        detener_tiempo = False
        inicio_tiempo = 0
        prueba = 1
        finalizar = 0

        # Establecemos el rango de colores que vamos a detectar
        #verdes_bajos = np.array([29, 86, 6], dtype=np.uint8)
        #verdes_altos = np.array([64, 255, 255], dtype=np.uint8)

        verdes_bajos = np.array([28,15,6 ], dtype=np.uint8)
        verdes_altos = np.array([100, 255, 255], dtype=np.uint8)
        
        verdes_bajos1 = np.array([40,60,6 ], dtype=np.uint8)
        verdes_altos1= np.array([100, 255, 255], dtype=np.uint8)
        
        verdes_bajos2 = np.array([30,30,6 ], dtype=np.uint8)
        verdes_altos2= np.array([100, 255, 255], dtype=np.uint8)


        # Iniciamos las cámaras

        cap = cv2.VideoCapture(0,cv2.CAP_V4L)

        cap.set(FRAME_PROP_WIDTH, frame_w)
        cap.set(FRAME_PROP_HEIGHT, frame_h)
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS,30)


        cap1 = cv2.VideoCapture(2,cv2.CAP_V4L)

        cap1.set(FRAME_PROP_WIDTH, frame_w)
        cap1.set(FRAME_PROP_HEIGHT, frame_h)
        cap1.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc(*"MJPG"))
        cap1.set(cv2.CAP_PROP_FPS,30)
        
        
        cap2 = cv2.VideoCapture(4,cv2.CAP_V4L)

        cap2.set(FRAME_PROP_WIDTH, frame_w)
        cap2.set(FRAME_PROP_HEIGHT, frame_h)
        cap2.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc(*"MJPG"))
        cap2.set(cv2.CAP_PROP_FPS,30)


        if(cap.isOpened() == False) or (cap1.isOpened() == False) or (cap2.isOpened() == False):
            print("ERROR: Camara no operativa")
            exit(-1)  # Error acceso a la camara
        while cap.isOpened() and cap1.isOpened() and cap2.isOpened():
            # Capturamos una imagen y la convertimos de RGB -> HSV
            
            _, imagen = cap.read()
            _, imagen1 = cap1.read()
            _, imagen2 = cap2.read()
           
        
            # Aplicar desenfoque para eliminar ruido
            frame = cv2.blur(imagen, (7, 7))
            frame1 = cv2.blur(imagen1, (7, 7))
            frame2 = cv2.blur(imagen2, (7, 7))

            # converimos la imagen a HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
            
            h, s, v = cv2.split(hsv)
            v -= 50
            hsv = cv2.merge((h, s, v))
            
            h, s, v = cv2.split(hsv1)
            v -= 50
            hsv1 = cv2.merge((h, s, v))
            
            h, s, v = cv2.split(hsv2)
            v -= 50
            hsv2 = cv2.merge((h, s, v))

            # Crear una mascara con solo los pixeles dentro del rango
            mask = cv2.inRange(hsv, verdes_bajos1, verdes_altos1)
            mask1 = cv2.inRange(hsv1, verdes_bajos, verdes_altos)
            mask2 = cv2.inRange(hsv2, verdes_bajos2, verdes_altos2)
            
         

            # Filtrar el ruido aplicando un OPEN seguido de un CLOSE
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3 , 3))
            
            eros = cv2.erode(mask, kernel, iterations=1)
            maskOpen = cv2.morphologyEx(eros, cv2.MORPH_OPEN, kernel)
            dila = cv2.dilate(maskOpen,None,iterations=1)
            maskClose = cv2.morphologyEx(dila, cv2.MORPH_CLOSE, kernel)
            maskClose = cv2.erode(maskClose, kernel, iterations=1)
            
            eros1 = cv2.erode(mask1, kernel, iterations=1)
            maskOpen1 = cv2.morphologyEx(eros1, cv2.MORPH_OPEN, kernel)
            dila1 = cv2.dilate(maskOpen1,None,iterations=1)
            maskClose1 = cv2.morphologyEx(dila1, cv2.MORPH_CLOSE, kernel)
            maskClose1 = cv2.erode(maskClose1, kernel, iterations=1)
            
            eros2 = cv2.erode(mask2, kernel, iterations=1)
            maskOpen2 = cv2.morphologyEx(eros2, cv2.MORPH_OPEN, kernel)
            dila2 = cv2.dilate(maskOpen2,None,iterations=1)
            maskClose2 = cv2.morphologyEx(dila2, cv2.MORPH_CLOSE, kernel)
            maskClose2 = cv2.erode(maskClose2, kernel, iterations=1)
            

      
            # Contornos de la imagen capturada
            maskFinal = maskClose
            cnts, h = cv2.findContours(
                maskFinal, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            maskFinal1 = maskClose1
            cnts1, h1 = cv2.findContours(
                maskFinal1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            maskFinal2 = maskClose2
            cnts2, h2 = cv2.findContours(
                maskFinal2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # Detectar el contorno mayor del area y especificar como cnt_best
            max_area = 0
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    cnt_best = cnt

            max_area1 = 0
            for cnt1 in cnts1:
                area1 = cv2.contourArea(cnt1)
                if area1 > max_area1:
                    max_area1 = area1
                    cnt_best1 = cnt1
                            
            max_area2 = 0
            for cnt2 in cnts2:
                area2 = cv2.contourArea(cnt2)
                if area2 > max_area2:
                    max_area2 = area2
                    cnt_best2 = cnt2
            #area1 = moments['m00']
            # área de la pantalla
            # print(area)
            if(max_area > 0 and max_area1 > 0) or (max_area2 > 0 and max_area1 > 0):
                
                if (max_area > 0  and max_area1 > 0):
                    # Encontrar el area de los objetos que detecta la camara
                    moments = cv2.moments(cnt_best)
                    moments1 = cv2.moments(cnt_best1)
                

                    # Buscamos el centro x, y del objeto
                    cx = int(moments['m10']/moments['m00'])
                    cy = int(moments['m01']/moments['m00'])
                    cx1 = int(moments1['m10']/moments1['m00'])
                    cy1 = int(moments1['m01']/moments1['m00'])
                
                    cv2.circle(imagen, (cx, cy), 7, (255, 0, 0), -1)
                    #cv2.circle(imagen1, (cx1, cy1), 7, (255, 0, 0), -1)
                

                    # Dibujamos un rectangulo como contorno de objeto
#                     x, y, w, h = cv2.boundingRect(cnt_best)
#                     x1, y1, w1, h1 = cv2.boundingRect(cnt_best1)
#                 
# 
#               
#                     cv2.rectangle(imagen, (x, y), (x+w, y+h),
#                               (0, 0, 255), 1, cv2.LINE_AA)
#                     cv2.putText(imagen, str(cx) + "," + str(cy),
#                             (cx, cy), font, 1, (0, 0, 55), 1)
# 
#                     cv2.rectangle(imagen1, (x1, y1), (x1+w1, y1+h1),
#                               (0, 0, 255), 1, cv2.LINE_AA)
#                     cv2.putText(imagen1, str(cx1) + "," + str(cy1),
#                             (cx1, cy1), font, 1, (0, 0, 55), 1)
                
              
                
                if (max_area2 > 0 and max_area1 > 0):
                    moments2 = cv2.moments(cnt_best2)
                    moments3 = cv2.moments(cnt_best1)
                    cx2 = int(moments2['m10']/moments2['m00'])
                    cy2 = int(moments2['m01']/moments2['m00'])
                    cx3 = int(moments3['m10']/moments3['m00'])
                    cy3 = int(moments3['m01']/moments3['m00'])
                    
                    cv2.circle(imagen2, (cx2, cy2), 7, (255, 0, 0), -1)
                    #cv2.circle(imagen1, (cx3, cy3), 7, (0, 0, 255), -1)
#                     x2, y2, w2, h2 = cv2.boundingRect(cnt_best2)
#                     x3, y3, w3, h3 = cv2.boundingRect(cnt_best3)
#                     cv2.rectangle(imagen, (x2, y2), (x2+w2, y2+h2),
#                               (0, 255,0), 1, cv2.LINE_AA)
#                     cv2.putText(imagen, str(cx2) + "," + str(cy2),
#                             (cx2, cy2), font, 1, (0, 0, 55), 1)
# 
#                     cv2.rectangle(imagen1, (x3, y3), (x3+w3, y3+h3),
#                               (0, 255, 0), 1, cv2.LINE_AA)
#                     cv2.putText(imagen1, str(cx3) + "," + str(cy3),
#                             (cx3, cy3), font, 1, (0, 0, 55), 1)
                
                     
                tot = 0
                tot1 = 0
            
            
                for i in coordenadas:
                   
                    cv2.circle(imagen2, (i[-2],i[-1]), 9, (0, 255, 255), 2)
                    #cv2.putText(imagen2,(str(i[-2])+"," + str(i[-1])),(i[-2]-5,i[-1]+70),font,0.5,(0, 0, 55),2)
                    if i[-2] >350:
                        tot+=1
                        cv2.putText(imagen2,str(tot),(i[-2]-5,i[-1]+40),font,0.5,(0, 0, 55),2)
                        if tot == 1:
                            cv2.putText(imagen2,"Inicio",(i[-2]-20,i[-1]+60),font,0.5,(0, 0, 55),2)
                    if i[-2] <350:
                        tot1+=1
                        cv2.putText(imagen2,str(tot1),(i[-2]-5,i[-1]+40),font,0.5,(0, 0, 55),2)
                        if tot1 == 6:
                            cv2.putText(imagen2,"Fin",(i[-2]-10,i[-1]+60),font,0.5,(0, 0, 55),2)
               
# 
#                 # Dibujar el punto de inicio y final que sirva como guia
# 
#                 # punto de inicio
#                 cv2.circle(imagen, (407, 240), 7, (0, 255, 255), 2)
#                 cv2.circle(imagen, (408, 240), 7, (0, 255, 255), 2)
#                 cv2.circle(imagen, (409, 240), 7, (0, 255, 255), 2)
#                 cv2.circle(imagen, (407, 241), 7, (0, 255, 255), 2)
#                 cv2.circle(imagen, (408, 241), 7, (0, 255, 255), 2)
#                 cv2.circle(imagen, (409, 241), 7, (0, 255, 255), 2)
#                 cv2.circle(imagen, (407, 242), 7, (0, 255, 255), 2)
#                 cv2.circle(imagen, (408, 242), 7, (0, 255, 255), 2)
#                 cv2.circle(imagen, (408, 242), 7, (0, 255, 255), 2)
# 
#                 # punto final
#                 cv2.circle(imagen, (177, 383), 7, (0, 153, 253), 2)
#                 cv2.circle(imagen, (178, 383), 7, (0, 153, 253), 2)
#                 cv2.circle(imagen, (179, 383), 7, (0, 153, 253), 2)
#                 cv2.circle(imagen, (177, 384), 7, (0, 153, 253), 2)
#                 cv2.circle(imagen, (178, 384), 7, (0, 153, 253), 2)
#                 cv2.circle(imagen, (179, 384), 7, (0, 153, 253), 2)
#                 cv2.circle(imagen, (177, 385), 7, (0, 153, 253), 2)
#                 cv2.circle(imagen, (178, 385), 7, (0, 153, 253), 2)
#                 cv2.circle(imagen, (179, 385), 7, (0, 153, 253), 2)
#                 
                
                #cv2.circle(imagen, (202, 57), 7, (0, 0, 255), 2)

                # Iniciar el recorrido
                if (lastX >= 0 and lastY >= 0 and cx >= 0 and cy >= 0) or (lastX2 >= 0 and lastY2 >= 0 and cx2 >= 0 and cy2 >= 0) or (lastX1 >= 0 and lastY1 >= 0 and cx1 >= 0 and cy1 >= 0) or (lastX3 >= 0 and lastY3 >= 0 and cx3 >= 0 and cy3 >= 0):
                    # Dibuja recorrido del objeto
                    #cv2.line(imagen, (cx, cy), (lastX, lastY), (0, 255, 0), 2)
                    #cv2.line(imagen1, (cx1, cy1), (lastX1, lastY1), (0, 255, 0), 2)
                    
                    if (cx2 ==coordenadas[0][0] and cy2==coordenadas[0][1]) or (cx2 ==coordenadas[0][0]+1 and cy2==coordenadas[0][1]) or (cx2 ==coordenadas[0][0]+2 and cy2==coordenadas[0][1]) or (cx2 ==coordenadas[0][0] and cy2==coordenadas[0][1]+1) or (cx2 ==coordenadas[0][0]+1 and cy2==coordenadas[0][1]+1) or (cx2 ==coordenadas[0][0]+2 and cy2==coordenadas[0][1]+1) or (cx2 ==coordenadas[0][0] and cy2==coordenadas[0][1]+2) or (cx2 ==coordenadas[0][0]+1 and cy2==coordenadas[0][1]+2) or (cx2 ==coordenadas[0][0]+2 and cy2==coordenadas[0][1]+2):
                    #if (cx2 ==coordenadas[1][0] and cy2==coordenadas[1][1]) or (cx2 ==coordenadas[1][0]+1 and cy2==coordenadas[1][1]) or (cx2 ==coordenadas[1][0]+2 and cy2==coordenadas[1][1]) or (cx2 ==coordenadas[1][0] and cy2==coordenadas[1][1]+1) or (cx2 ==coordenadas[1][0]+1 and cy2==coordenadas[1][1]+1) or (cx2 ==coordenadas[1][0]+2 and cy2==coordenadas[1][1]+1) or (cx2 ==coordenadas[1][0] and cy2==coordenadas[1][1]+2) or (cx2 ==coordenadas[1][0]+1 and cy2==coordenadas[1][1]+2) or (cx2 ==coordenadas[1][0]+2 and cy2==coordenadas[1][1]+2):
                    #if (cx2 ==coordenadas[2][0] and cy2==coordenadas[2][1]) or (cx2 ==coordenadas[2][0]+1 and cy2==coordenadas[2][1]) or (cx2 ==coordenadas[2][0]+2 and cy2==coordenadas[2][1]) or (cx2 ==coordenadas[2][0] and cy2==coordenadas[2][1]+1) or (cx2 ==coordenadas[2][0]+1 and cy2==coordenadas[2][1]+1) or (cx2 ==coordenadas[2][0]+2 and cy2==coordenadas[2][1]+1) or (cx2 ==coordenadas[2][0] and cy2==coordenadas[2][1]+2) or (cx2 ==coordenadas[2][0]+1 and cy2==coordenadas[2][1]+2) or (cx2 ==coordenadas[2][0]+2 and cy2==coordenadas[2][1]+2):
 
                        cv2.circle(imagen2, (coordenadas[0][0],coordenadas[0][1]), 9, (0, 153, 255), 2)
                        inicio = 1
                        fin = 0
                        detener = 0

                        if inicio_tiempo == 0:
                            start_time = time()
                            inicio_tiempo = 1

                    if (cx2 ==coordenadas[-1][0] and cy2==coordenadas[-1][1]) or (cx2 ==coordenadas[-1][0]+1 and cy2==coordenadas[-1][1]) or (cx2 ==coordenadas[-1][0]+2 and cy2==coordenadas[-1][1]) or (cx2 ==coordenadas[-1][0] and cy2==coordenadas[-1][1]+1) or (cx2 ==coordenadas[-1][0]+1 and cy2==coordenadas[-1][1]+1) or (cx2 ==coordenadas[-1][0]+2 and cy2==coordenadas[-1][1]+1) or (cx2 ==coordenadas[-1][0] and cy2==coordenadas[-1][1]+2) or (cx2 ==coordenadas[-1][0]+1 and cy2==coordenadas[-1][1]+2) or (cx2 ==coordenadas[-1][0]+2 and cy2==coordenadas[-1][1]+2):
                    #if (cx2 ==coordenadas[-2][0] and cy2==coordenadas[-2][1]) or (cx2 ==coordenadas[-2][0]+1 and cy2==coordenadas[-2][1]) or (cx2 ==coordenadas[-2][0]+2 and cy2==coordenadas[-2][1]) or (cx2 ==coordenadas[-2][0] and cy2==coordenadas[-2][1]+1) or (cx2 ==coordenadas[-2][0]+1 and cy2==coordenadas[-2][1]+1) or (cx2 ==coordenadas[-2][0]+2 and cy2==coordenadas[-2][1]+1) or (cx2 ==coordenadas[-2][0] and cy2==coordenadas[-2][1]+2) or (cx2 ==coordenadas[-2][0]+1 and cy2==coordenadas[-2][1]+2) or (cx2 ==coordenadas[-2][0]+2 and cy2==coordenadas[-2][1]+2):
                    
                        cv2.circle(imagen2, (coordenadas[-1][0],coordenadas[-1][1]), 9, (0, 153, 255), 2)
                        inicio = 0
                        fin = 1

                    # Condición para guardar datos del recorrido y del tiempo
                    # Cuando se ha llegado al inicio pero falta llegar al final del recorrido,
                    # dependiendo si ya se han cumplido la cantidad de repeticiones

                    if inicio == 1 and fin == 0 and prueba <= int(pack.get()):

                        # Mostramos sus coordenadas por pantalla
                        print("cx = ", cx)
                        print("cy = ", cy)
                        print("cx1 = ", cx1)
                        print("cy1 = ", cy1)
                        print("cx2 = ", cx2)
                        print("cy2 = ", cy2)
                        print("cx3 = ", cx3)
                        print("cy3 = ", cy3)

                        # Mensaje al iniciar la repetición

                        cv2.circle(imagen2, (25, 15), 8, (0, 0, 255), -1)
                        cv2.putText(imagen2, "Finalizar", (35, 20),
                                    font, 0.6, (0, 0, 255), 2, 2)
                        cv2.putText(imagen2, "Esc-Salir", (125, 20),
                                    font, 0.6, (0, 255, 0), 2, 2)

#                         cv2.circle(imagen1, (25, 15), 8, (0, 0, 255), -1)
#                         cv2.putText(imagen1, "Finalizar", (35, 20),
#                                     font, 0.6, (0, 0, 255), 2, 2)
#                         cv2.putText(imagen1, "Esc-Salir", (125, 20),
#                                     font, 0.6, (0, 255, 0), 2, 2)

                        # Abrir archivo para edición
                        #fo = open(e1.get() + '-Ejercicio' + "/" + e1.get() + "-Camara1-" +
                        #          "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                        #fo.write(str(cx) + "," + str(cy) + "\n")

                        #fo = open(e1.get() + '-Ejercicio' + "/" + e1.get() + "-Camara2-" +
                        #          "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                        #fo.write(str(cx1) + "," + str(cy1) + "\n")
                        
                        #Conversión de coordenadas de Pixeles a Centimetros
                        x, y, z = conversion_1p(cx, cy, cx1, cy1)
                        
                        #Se almacenan las coordenadas pixel en un archivo csv
                        fo = open(e1.get() + '-Ejercicio-3-1' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" +
                                  "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                        fo.write(str(cx) + "," + str(cy) + "," + str(cy1) + "\n")
                        
                        #Se almacenan las coordenadas centimentro en un archivo csv
                        f1 = open(e1.get() + '-Ejercicio-3-1' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" +
                                  "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + "-cm.csv", "a")
                        f1.write(str(x) + "," + str(y) + "," + str(z) + "\n")

                        # Calculo de tiempo de recorrido
                        elapsed_time = time() - start_time
                        minutos = int(float(elapsed_time)) / 60
                        segundos = int(float(elapsed_time)) % 60
                        d = decimal.Decimal(elapsed_time)
                        decimal.getcontext().prec = 4

                    else:

                        # Si finaliza el recorrido y prueba es menor o igual a las repeticiones solicitadas
                        # realiza la petición de inicio en pantalla

                        if finalizar == 0 and prueba <= int(pack.get()):

                            cv2.circle(imagen2, (25, 15), 8, (232, 189, 30), -1)
                            cv2.putText(imagen2, "Iniciar Repeticion " + str(prueba),
                                        (35, 20), font, 0.6, (232, 189, 30), 2, 2)
# 
#                             cv2.circle(imagen1, (25, 15), 8,
#                                        (232, 189, 30), -1)
#                             cv2.putText(imagen1, "Iniciar Repeticion " + str(prueba),
#                                         (35, 20), font, 0.6, (232, 189, 30), 2, 2)
                            
                           
                            
            

                        # Sino se cumple la condición, se envia en pantalla salir del ejercicio
                        else:
                            cv2.putText(imagen2, "Fin de Repeticiones",
                                        (35, 20), font, 0.6, (232, 189, 30), 2, 2)
                            cv2.putText(imagen2, "Esc-Salir", (230, 20),
                                        font, 0.6, (0, 255, 0), 2, 2)

#                             cv2.putText(imagen1, "Fin de Repeticiones",
#                                         (35, 20), font, 0.6, (232, 189, 30), 2, 2)
#                             cv2.putText(imagen1, "Esc-Salir",
#                                         (230, 20), font, 0.6, (0, 255, 0), 2, 2)
#                             
                            

                    # Si se finaliza el recorrido, se detiene el conteo de tiempo y almacena el tiempo final en TF
                    # A su vez, se guarda en el archivo de texto antes abierto en donde se guardaron las coordenadas de recorrido
                    if fin == 1:
                        if detener == 0:
                            ruta = e1.get() + '-Ejercicio-3-1' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                            ruta0 = e1.get() + '-Ejercicio-3-1' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")+"-cm"
                            ruta1 = e1.get() + '-Ejercicio-3-1' + "/" + "Maps" + "/" + e1.get() + "-Maps-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                            ruta2 = e1.get() + '-Ejercicio-3-1' + "/" + "Graficas" + "/" + e1.get() + "-Grafica-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                            #rint(ruta)
                            
                            elapsed_time = time() - start_time
                            #fo = open(e1.get() + '-Ejercicio' + "/" + e1.get() + "-Camara1-" + "Rep" + str(
                            #    prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                            #fo.write("TF" + "," + str(elapsed_time) + "\n")
                            #fo.close()

                            #fo = open(e1.get() + '-Ejercicio' + "/" + e1.get() + "-Camara2-" + "Rep" + str(
                            #    prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                            #fo.write("TF" + "," + str(elapsed_time) + "\n")
                            #fo.close()
                            
                            #Se guarda el tiempo de ejecución del ejercicio en los archivos de datos
                            fo = open(e1.get() + '-Ejercicio-3-1' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" + "Rep" + str(
                                prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                            fo.write("TF" + "," + str(elapsed_time) + "," +"0"+ "\n")
                            fo.close()
                            
                            f1 = open(e1.get() + '-Ejercicio-3-1' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" + "Rep" + str(
                                prueba) + "-" + tiempo.strftime("%d-%m-%Y") + "-cm.csv", "a")
                            f1.write("TF" + "," + str(elapsed_time) + "," +"0"+ "\n")
                            f1.close()

                            print("Tiempo Final: ", str(elapsed_time))
                            
                            if ruta != "":
                                
                                #ruta_nueva = normalizar(ruta)
                                maps(ruta0, ruta1)
                                grafica(ruta, ruta2, prueba)

                            detener = 1

                         # Mientras las repeticiones de la prueba no terminen, van incrementando
                         # Se reinicia el tiempo cada que se cumple con una repeticion (inicio-final)
                            if prueba <= int(pack.get()):
                                inicio = 0
                                fin = 0
                                detener_tiempo = False
                                inicio_tiempo = 0
                                prueba = prueba + 1
                         # Si finalizar es igual a 1, quiere decir que a terminado el recorrido
                            else:
                                finalizar = 1

                lastX, lastY = cx, cy
                lastX1, lastY1 = cx1, cy1
                lastX2, lastY2 = cx2, cy2
                lastX3, lastY3 = cx3, cy3
               

                # Mostramos la imagen original con la marca del centro y
                # la mascara

                #cv2.imshow('Mascara1', maskClose)
                #cv2.imshow('brillo', hsv)
                #cv2.imshow('Difuminado', frame)
                #cv2.imshow('Mascara2', maskClose2)
                #cv2.imshow('Mascara3', maskClose1)
                #cv2.imshow('Camara1_Maestra', imagen)
                cv2.imshow('Laparoscopia: Ejercicio 3-1', imagen2)
                #cv2.imshow('Camara2_Esclava', imagen1)

                tecla = cv2.waitKey(100) & 0xFF
                if tecla == 27:
                    cap.release()
                    cap1.release()
                    cap2.release()
                    cv2.destroyAllWindows()
                    break


master = Tk()
master.title("Entrenamiento")
master.resizable(0, 0)
if sys.platform == "win32":
    master.geometry("300x220")
else:
    master.geometry("350x220")
master.config(bg="#ededed")

center(master)

Label(master, text="Traslado físico de objetos \ncon 1 pinza", bg='#ededed').grid(
    row=0, column=1, columnspan=2)

Label(master, text="Usuario:", bg='#ededed').grid(row=2, column=0)
e1 = Entry(master, width=24, bg='#FFFFFF')
e1.grid(row=2, column=1)

Label(master, text="Repeticiones:", bg='#ededed').grid(row=3, column=0)

pack = ttk.Combobox(master, state="readonly", values=('3', '4', '5'))
pack.grid(row=3, column=1, columnspan=2, sticky=W, pady=4)
pack.set("3")

Button(master, text='Iniciar', cursor="hand2", fg='Black', bg='#f7c282', width=20, command=lambda: createDir(
    e1.get() + '-Ejercicio-3-1')).grid(row=4, column=1, columnspan=2, sticky=W, pady=4)
Button(master, text='Analizar', cursor="hand2", fg='Black', bg='#31b6fd', width=20, command=abrir).grid(row=5, column=1, columnspan=2, sticky=W, pady=4)

Button(master, text='Salir', cursor="hand2", fg='White', bg='#bb2003',
       width=20, command=master.destroy).grid(row=6, column=1, columnspan=2, sticky=W, pady=4)

master.mainloop()
