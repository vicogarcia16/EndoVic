#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import time as tiempo
from time import time
import sys
import os.path
import random
import decimal
import rangos
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from math import *
from scipy import signal

font = cv2.FONT_HERSHEY_DUPLEX
color = (0, 0, 0)

#Ancho y alto del video que se desea capturar:
# 160.0 x 120.0, 176.0 x 144.0, 320.0 x 240.0, 352.0 x 288.0, 640.0 x 480.0
# 800.00 x 600.0, 1280.0 x 720.0

frame_w = 640
frame_h = 480

global iniciar, posiciones1, posiciones2, n_elementos
posiciones1 = []
posiciones2 = []
iniciar = 0
n_elementos = 15

def abrir():
    if e1.get() != "":
        rangos.principal(e1.get() + '-Ejercicio-2')
    else:
        messagebox.showwarning(
                "Advertencia", "No ha indicado un nombre de usuario.")

def createDir(directorio):
    if os.path.exists(directorio):
        messagebox.showwarning(
            "Advertencia", "El nombre de usuario ya existe.")
    else:
        if e1.get() != "":
            os.mkdir(directorio)
            os.mkdir(directorio +'/Repeticiones')
            #os.mkdir(directorio +'/Maps')
            os.mkdir(directorio +'/Graficas')
            os.mkdir(directorio +'/Scores')
            camara()
        else:
            messagebox.showwarning(
                "Advertencia", "No ha indicado un nombre de usuario.")


def tarea2(dibujar, posicion_inicial, posicion_final):
    
    global iniciar, tarea_dos, posiciones1, posiciones2, n_elementos
  
    tarea_dos = create_blank(frame_w, frame_h, rgb_color=color)
    
    if e1.get() == "":
        messagebox.showwarning("Advertencia", "No ha indicado un nombre de usuario.")
        iniciar = 1
    else:    
        if dibujar == 1:
            posiciones1 = []
            posiciones2 = []
            iniciar = 0

            for i in range(0,n_elementos):
                posiciones1.append(str(random.randint(55, 270)) + "," + str(random.randint(60, 350)))
                posiciones2.append(str(random.randint(315, 600)) + "," + str(random.randint(60, 400)))

            for i in range(0,n_elementos):
                xy1 = posiciones1[i].split(',')
                cv2.circle(tarea_dos, (int(xy1[0]),int(xy1[1])), 10, (255,0,0), -1)
                cv2.putText(tarea_dos, str(i), (int(xy1[0]),int(xy1[1])), font, 0.6, (255,255,255), 0, 4)

                xy2 = posiciones2[i].split(',')
                cv2.circle(tarea_dos, (int(xy2[0]),int(xy2[1])), 10, (0,0,255), -1)
                cv2.putText(tarea_dos, str(i), (int(xy2[0]),int(xy2[1])), font, 0.6, (255,255,255), 0, 4)
        else:
            if posicion_inicial != -1:
                del posiciones1[posicion_inicial]

            n1 = len(posiciones1)
         

            if posicion_final != -1:
                del posiciones2[posicion_final]

            n2 = len(posiciones2)

            for i in range(0,n1):
                xy1 = posiciones1[i].split(',')
                cv2.circle(tarea_dos, (int(xy1[0]),int(xy1[1])), 10, (255,0,0), -1)
                cv2.putText(tarea_dos, str(i), (int(xy1[0]),int(xy1[1])), font, 0.6, (255,255,255), 0, 4)

            for i in range(0,n2):
                xy2 = posiciones2[i].split(',')
                cv2.circle(tarea_dos, (int(xy2[0]),int(xy2[1])), 10, (0,0,255), -1)
                cv2.putText(tarea_dos, str(i), (int(xy2[0]),int(xy2[1])), font, 0.6, (255,255,255), 0, 4)

            
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
        x[i]=(x[i]*0.042)/100
        y[i]=(y[i]*0.042)/100
        z[i]=(z[i]*0.042)/100
        
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
    fo.write(t1 + "," + str(PLD) + "," + str(DPD) + "," + "0" + "," + str(MS1) + "," + str(MS2).strip('[]') + "," + str(Mean_SpeedD) + "," + str(Mean_AccD) + "," + str(idleD) + "," + str(A_PLD) + "," + str(A_VD)  + "," + str(EndoEAD) + "," + str(EndoEVD) + "," + "0" + "\n")
    fo.close
                
def rotateImage(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
 
    # rotate the image by 180 degrees
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def show_entry_fields():
   print("Nombre del archivo: %s" % (e1.get()))

def static_pos(posX,posY):
    static_pos.cx = posX
    static_pos.cy = posY
    return static_pos.cx, static_pos.cy
   
def center(ventana):
    ventana.update_idletasks()
    w=ventana.winfo_width()
    h=ventana.winfo_height()
    extraW=ventana.winfo_screenwidth()-w
    extraH=ventana.winfo_screenheight()-h
    ventana.geometry("%dx%d%+d%+d" % (w,h,extraW/2,extraH/2))

def create_blank(width, height, rgb_color=(0, 0, 255)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def camara():
    
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
    
    bandera = 0
    
    start_time = 0
    elapsed_time = 0
    minutos = 0
    segundos = 0
    
    minuto = int(float(e2.get())) / 60
    segundo = int(float(e2.get())) % 60 
   
    #graficar(0)
    tarea2(1,-1,-1)
    
    if iniciar != 1:
        # Indices de las propiedades de video (no editar)
        FRAME_PROP_WIDTH = 3
        FRAME_PROP_HEIGHT = 4
       
        # Definir rango de color a identificar (HSV)
        verdes_bajos = np.array([28,15,6 ], dtype=np.uint8)
        verdes_altos = np.array([100, 255, 255], dtype=np.uint8)
        
        verdes_bajos1 = np.array([40,45,6 ], dtype=np.uint8)
        verdes_altos1= np.array([100, 255, 255], dtype=np.uint8)
        
        verdes_bajos2 = np.array([30,19,6 ], dtype=np.uint8)
        verdes_altos2= np.array([100, 255, 255], dtype=np.uint8)
        
        # Iniciar captura de video con el tamano deseado
        cap = cv2.VideoCapture(4, cv2.CAP_V4L)
        
        cap.set(FRAME_PROP_WIDTH, frame_w)
        cap.set(FRAME_PROP_HEIGHT, frame_h)
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS,30)

        cap1 = cv2.VideoCapture(2, cv2.CAP_V4L)

        cap1.set(FRAME_PROP_WIDTH, frame_w)
        cap1.set(FRAME_PROP_HEIGHT, frame_h)
        cap1.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc(*"MJPG"))
        cap1.set(cv2.CAP_PROP_FPS,30)
        
        cap2 = cv2.VideoCapture(0,cv2.CAP_V4L)

        cap2.set(FRAME_PROP_WIDTH, frame_w)
        cap2.set(FRAME_PROP_HEIGHT, frame_h)
        cap2.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc(*"MJPG"))
        cap2.set(cv2.CAP_PROP_FPS,30)
       
        inicio = 0
        fin = 0
        detener = 0
        detener_tiempo = False
        encontrar = 0
        score = 0
        contador = 0
        start = 0
        iniciar_tiempo = 0
        cerrar_archivo = 0
        abrir_archivo = 0
        guardar_imagen = 1
        prueba = 1
       
        if(cap.isOpened() == False) or (cap1.isOpened() == False) or (cap2.isOpened() == False):
            print("ERROR: Camara no operativa")
            exit(-1)  # Error acceso a la camara
        while cap.isOpened() and cap1.isOpened() and cap2.isOpened():
            if start == 1 and iniciar_tiempo == 0:
                start_time = time()
                start = 0
                iniciar_tiempo = 1
            
            if iniciar_tiempo == 1:
                elapsed_time = time() - start_time
            
            minutos = int(float(elapsed_time)) / 60
            segundos = int(float(elapsed_time)) % 60

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
            mask = cv2.inRange(hsv, verdes_bajos2, verdes_altos2)
            mask1 = cv2.inRange(hsv1, verdes_bajos, verdes_altos)
            mask2 = cv2.inRange(hsv2, verdes_bajos1, verdes_altos1)

            # Filtrar el ruido aplicando un OPEN seguido de un CLOSE
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3 , 3))
            
            eros = cv2.erode(mask, kernel, iterations=2)
            maskOpen = cv2.morphologyEx(eros, cv2.MORPH_OPEN, kernel)
            dila = cv2.dilate(maskOpen,None,iterations=1)
            maskClose = cv2.morphologyEx(dila, cv2.MORPH_CLOSE, kernel)
            maskClose = cv2.erode(maskClose, kernel, iterations=2)
            
            eros1 = cv2.erode(mask1, kernel, iterations=2)
            maskOpen1 = cv2.morphologyEx(eros1, cv2.MORPH_OPEN, kernel)
            dila1 = cv2.dilate(maskOpen1,None,iterations=1)
            maskClose1 = cv2.morphologyEx(dila1, cv2.MORPH_CLOSE, kernel)
            maskClose1 = cv2.erode(maskClose1, kernel, iterations=2)
            
            eros2 = cv2.erode(mask2, kernel, iterations=2)
            maskOpen2 = cv2.morphologyEx(eros2, cv2.MORPH_OPEN, kernel)
            dila2 = cv2.dilate(maskOpen2,None,iterations=1)
            maskClose2 = cv2.morphologyEx(dila2, cv2.MORPH_CLOSE, kernel)
            maskClose2 = cv2.erode(maskClose2, kernel, iterations=2)



            # Contornos de la imagen capturada
            maskFinal = maskClose
            cnts, h = cv2.findContours( maskFinal, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            maskFinal1 = maskClose1
            cnts1, h1 = cv2.findContours(maskFinal1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            maskFinal2 = maskClose2
            cnts2, h2 = cv2.findContours(maskFinal2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
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
            # Ejecutar este bloque solo si se encontro un area

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
                
                    #cv2.circle(imagen, (cx, cy), 7, (255, 0, 0), -1)
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

 
            
                if bandera == 1:
                   
                    cv2.circle(imagen, (cx, cy), 10, (255,0,0), -1)
                    cv2.putText(imagen, str(posicion), (cx,cy), font, 0.6, (255,255,255), 0, 4)
                    fo = open(e1.get()  + '-Ejercicio-2' + "/" + "Scores" + "/" + e1.get() + "-Score-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a+")
                   
                    #--fo.write(str(cx) + "," + str(cy) + "\n")
                    fo.close()
                    f1 = open(e1.get() + '-Ejercicio-2' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" +
                              "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                    f1.write(str(cx2) + "," + str(cy2) + "," + str(cy3) + "\n")
                    
                    # Dibujar un rectangulo alrededor del objeto
                    #x, y, w, h = cv2.boundingRect(best_cnt)
                    #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if (lastX >= 0 and lastY >= 0 and cx >= 0 and cy >= 0) or (lastX2 >= 0 and lastY2 >= 0 and cx2 >= 0 and cy2 >= 0) or (lastX1 >= 0 and lastY1 >= 0 and cx1 >= 0 and cy1 >= 0) or (lastX3 >= 0 and lastY3 >= 0 and cx3 >= 0 and cy3 >= 0):
                    
                    for i in range(0,len(posiciones1)):
                        puntos = posiciones1[i].split(',')
                        punto = str(cx) + "," + str(cy)
                        if( (punto == posiciones1[0]) or (cx == int(puntos[0]) and cy == int(puntos[1]) - 2 ) or ( cx == int(puntos[0])  + 2 and cy == int(puntos[1])) or (cx == int(puntos[0]) and cy == int(puntos[1])  + 2) or (cx == int(puntos[0])  - 2 and cy == int(puntos[1]))):
                            cv2.circle(tarea_dos, (cx, cy), 10, (0,255,0), -1)
                            bandera = 1
                            posicion = i
                            encontrar = 1
                            start = 1
                    
                    if encontrar == 1:
                        puntos2 = posiciones2[posicion].split(',')
                        punto2 = str(cx) + "," + str(cy)
                        
                        if( (punto2 == posiciones2[0]) or (cx == int(puntos2[0]) and cy == int(puntos2[1]) - 2 ) or ( cx == int(puntos2[0])  + 2 and cy == int(puntos2[1])) or (cx == int(puntos2[0]) and cy == int(puntos2[1])  + 2) or (cx == int(puntos2[0])  - 2 and cy == int(puntos2[1]))):
                            cv2.circle(tarea_dos, (cx, cy), 10, (0,255,0), -1)
                            bandera = 0
                            tarea2(0,posicion,posicion)
                            encontrar = 0
                            score += 1
                            contador += 1

                lastX, lastY = cx, cy
                lastX1, lastY1 = cx1, cy1
                lastX2, lastY2 = cx2, cy2
                lastX3, lastY3 = cx3, cy3
                


           # Mostrar la imagen original con todos los overlays
        
            if minuto == minutos and segundo == segundos:
                cv2.putText(imagen, "Tiempo Finalizado!", (35, 20), font, 0.6, (0,0,0), 2, 1)
                cv2.putText(imagen, "Puntaje: " + str(score), (35, 60), font, 0.6, (0,0,0), 2, 1)
                detener_tiempo = True
                bandera = 0
                if cerrar_archivo == 0:
                    ruta = e1.get() + '-Ejercicio-2' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                    ruta1 = e1.get() + '-Ejercicio-2' + "/" + "Maps" + "/" + e1.get() + "-Maps-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                    ruta2 = e1.get() + '-Ejercicio-2' + "/" + "Graficas" + "/" + e1.get() + "-Grafica-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                    
                    fo = open(e1.get()  + '-Ejercicio-2' + "/" + "Scores" + "/" + e1.get() + "-Score-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                    fo.write("Aciertos" + "," + str(score) + "\n")
                    fo.close()
                    
                    f1 = open(e1.get() + '-Ejercicio-2' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" +
                              "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                    f1.write("TF" + "," + str(elapsed_time) + "," +"0"+ "\n")
                    f1.close()
                    cerrar_archivo = 1
                    
                    if ruta != "":
                        #ruta_nueva = normalizar(ruta)
                        grafica(ruta, ruta2, prueba)
                        #maps(ruta, ruta1)
            else:
                if detener_tiempo == True:
                    cv2.putText(imagen, "Tiempo Finalizado!", (35, 20), font, 0.6, (0,0,0), 2, 1)
                    cv2.putText(imagen, "Puntaje: " + str(score), (35, 60), font, 0.6, (0,0,0), 2, 1)
                    bandera = 0
                       
                    if prueba != int(pack.get()):
                        prueba = prueba + 1
                        detener_tiempo = False
                        start = 0
                        iniciar_tiempo = 0
                        cerrar_archivo = 0
                        score = 0
                        bandera = 0
                        start_time = 0
                        elapsed_time = 0
                        minutos = 0
                        segundos = 0
                        contador = 0
                        tarea2(1,-1,-1)
                else:
                    if n_elementos == contador or detener_tiempo == True:
                        cv2.putText(imagen, "Puntaje: " + str(score), (35, 60), font, 0.6, (0,0,0), 2, 1)
                        bandera = 0
                        if cerrar_archivo == 0:
                            ruta = e1.get() + '-Ejercicio-2' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                            ruta1 = e1.get() + '-Ejercicio-2' + "/" + "Maps" + "/" + e1.get() + "-Maps-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                            ruta2 = e1.get() + '-Ejercicio-2' + "/" + "Graficas" + "/" + e1.get() + "-Grafica-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                    
                            fo = open(e1.get()  + '-Ejercicio-2' + "/" + "Scores" + "/" + e1.get() + "-Score-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                            fo.write("Aciertos" + "," + str(score) + "\n")
                            fo.close()
                            
                            f1 = open(e1.get() + '-Ejercicio-2' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" +
                              "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                            f1.write("TF" + "," + str(elapsed_time) + "," +"0"+ "\n")
                            f1.close()
                            cerrar_archivo = 1
                            
                            if ruta != "":
                                #ruta_nueva = normalizar(ruta)
                                grafica(ruta, ruta2, prueba)
                                #maps(ruta, ruta1)
                    else:
                        cv2.putText(imagen,"Repeticion " + str(prueba), (35, 20), font, 0.6, (0, 0, 0), 2, 1)
                        d = decimal.Decimal(elapsed_time)
                        decimal.getcontext().prec = 4
                        cv2.putText(imagen, "Tiempo: " + str(minutos) + " minuto(s), " + str(segundos) + " segundo(s).", (35, 40), font, 0.6, 244, 2, 8)
                        tiempo_realizado = d * 1
                        cv2.putText(imagen, "Tiempo: " + str(tiempo_realizado), (35, 60), font, 0.6, (0,0,0), 2, 1)
                        cv2.addWeighted(imagen,0.7,tarea_dos,1.0,0.3, imagen)
                    
            
            # perform the actual resizing of the image and show it
            #------resized = cv2.resize(img, (1366,768), interpolation = cv2.INTER_AREA)    
            cv2.imshow("Laparoscopia: Ejercicio 2", imagen)
            #cv2.imshow("m1", maskClose)
            #cv2.imshow("m2", maskClose1)
            #cv2.imshow("m3", maskClose2)
            #cv2.imshow("Laparoscopia: Ejercicio 2qw", maskClose)
         
            # Salir del bucle si se presiona ESC
            k = cv2.waitKey(100) & 0xFF
            if k == 27:
                # Limpieza y fin de programa
                cap.release()
                cap1.release()
                cap2.release()
                cv2.destroyAllWindows()
                break
         



master = Tk()
master.title("Entrenamiento")
master.resizable(0,0)
if sys.platform == "win32":
    master.geometry("300x220")
else:
    master.geometry("350x220")
master.config(bg="#ededed")

center(master)

Label(master,text="Transferencia virtual", bg='#ededed').grid(row=0,column=1,columnspan=2)

Label(master, text="Usuario:", bg='#ededed').grid(row=2, column=0)
e1 = Entry(master, width=23, bg='#FFFFFF')
e1.grid(row=2, column=1)

Label(master, text="Repeticiones:", bg='#ededed').grid(row=3, column=0)

pack = ttk.Combobox(master, state="readonly", values=('3','4','5','6'))
pack.grid(row=3, column=1,columnspan=2, sticky=W, pady=4)
pack.set("3")

Label(master, text="Tiempo:", bg='#ededed').grid(row=4, column=0)
e2 = Entry(master, width=23, bg='#EEE')
e2.grid(row=4, column=1)
e2.insert(0, "80")
Label(master, text="seg.", bg='#ededed').grid(row=4, column=3)

Button(master, text='Iniciar',cursor="hand2", fg='Black', bg='#f7c282', width=20, command=lambda:createDir(e1.get() + '-Ejercicio-2')).grid(row=5, column=1,columnspan=2, sticky=W, pady=4)
Button(master, text='Analizar',cursor="hand2", fg='Black', bg='#31b6fd', width=20, command=abrir).grid(row=6, column=1, columnspan=2, sticky=W, pady=4)

Button(master, text='Salir', cursor="hand2", fg='White', bg='#bb2003', width=20, command=exit).grid(row=7, column=1, columnspan=2, sticky=W, pady=4)

mainloop()