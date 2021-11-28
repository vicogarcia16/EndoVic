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
from matriz_conversion import conversion_2p

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
        directory = e1.get() + '-Ejercicio-3-2'
       
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
            coordenadas.append((int(row[0]),int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[5])))
            
    for i in coordenadas:
        if i not in nueva:
            fo= open(ruta_nueva + ".csv", "a")
            fo.write(str(i[0])+","+ str(i[1])+ ","+str(i[2])+ ","+str(i[3])+ ","+str(i[4])+ ","+str(i[5]) + "\n")
            nueva.append((i[0],i[1],i[2],i[3],i[4],i[5]))
            
    reader = csv.reader(open(ruta +".csv", "r"))
    for index,row in enumerate(reader):
        if row[0] == 'TF':
            fo= open(ruta_nueva + ".csv", "a")
            fo.write(row[0]+","+ row[1]+","+row[2]+","+row[3]+ ","+row[4]+ ","+row[5]+  "\n")

    return ruta_nueva

def grafica(ruta, ruta2, prueba):
    
    X = np.genfromtxt(ruta + '.csv', delimiter=',')
    x = X[0:-1:,0]
    y = X[0:-1:,1]
    z = X[0:-1:,2]
    x1 = X[0:-1:,3]
    y1 = X[0:-1:,4]
    z1 = X[0:-1:,5]

    fig = plt.figure(figsize=(8,6))
    # Creamos el plano 3D
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.plot(x,z,y, linestyle='-', color = 'black', linewidth = 1.5)
    ax1.view_init(-165,-65)
    #plt.subplots_adjust(left = 0.05)
    #ax1.plot_wireframe(x2, y2, y22)
    plt.title('Gráfica de movimiento del usuario (mano derecha)- Repetición '+str(prueba))
    ax1.set_xlabel('eje x')
    ax1.set_ylabel('eje z')
    ax1.set_zlabel('eje y')
    plt.savefig(ruta2 + '_derecha.png')
    
    fig1 = plt.figure(figsize=(8,6))
    # Creamos el plano 3D
    ax2 = fig1.add_subplot(111, projection='3d')

    ax2.plot(x1,z1,y1, linestyle='-', color = 'black', linewidth = 1.5)
    ax2.view_init(-165,-65)
    #plt.subplots_adjust(left = 0.05)
    #ax1.plot_wireframe(x2, y2, y22)
    plt.title('Gráfica de movimiento del usuario (mano izquierda)- Repetición '+str(prueba))
    ax2.set_xlabel('eje x')
    ax2.set_ylabel('eje z')
    ax2.set_zlabel('eje y')
    plt.savefig(ruta2 + '_izquierda.png')
    #plt.show()

def maps(ruta, ruta1):
    X = np.genfromtxt(ruta + '.csv', delimiter=',')
    xa = X[0:-1:,0]
    ya = X[0:-1:,1]
    za = X[0:-1:,2]
    
    xb = X[0:-1:,3]
    yb = X[0:-1:,4]
    zb = X[0:-1:,5]
    
    t = X[-1:,1]
    t1 = t[-1:1]
    t1 = str(t).strip('[]')
    
    #Cálculo de MAP's"
    for i in range(len(xa)):
        xa[i]=(xa[i])/100
        ya[i]=(ya[i])/100
        za[i]=(za[i])/100
        xb[i]=(xb[i])/100
        yb[i]=(yb[i])/100
        zb[i]=(zb[i])/100
        
    #EndoViS Path Length
    #Derecha
    dist = np.sqrt(np.diff(xa,1)**2 + np.diff(ya,1)**2 + np.diff(za,1)**2)
    PLD=np.sum(dist)
    
    #Izquierda
    dist1 = np.sqrt(np.diff(xb,1)**2 + np.diff(yb,1)**2 + np.diff(zb,1)**2)
    PLI=np.sum(dist1)
    

    #EndoViS Depth Perception
    #Derecha
    dist = np.sqrt(np.diff(ya,1)**2 + np.diff(za,1)**2)
    DPD=np.sum(dist)
    
    #izquierda
    dist1 = np.sqrt(np.diff(yb,1)**2 + np.diff(zb,1)**2)
    DPI=np.sum(dist1)

    #EndoViS Motion Smoothness
    #Derecha
    MSD = np.sum((np.diff(xa,3)**2 + np.diff(ya,3)**2 + np.diff(za,3)**2))
    MS1=np.sqrt(0.5*(MSD))
    #Carvalo y EVA
    cte = (t**5)/(2*PLD**2)
    MS2 = np.sqrt(cte*(MSD))
    
    #izquierda
    MSI = np.sum((np.diff(xb,3)**2 + np.diff(yb,3)**2 + np.diff(zb,3)**2))
    MS3=np.sqrt(0.5*(MSI))
    #Carvalo y EVA
    cte = (t**5)/(2*PLD**2)
    MS4 = np.sqrt(cte*(MSI))



    # Resampleo de la señal a cada segundo
    num = round(len(xa)/30)
    f = round(len(xa)/num)
    
    xxa = signal.resample_poly(xa,1,f,window = ('kaiser',3.2))
    yya = signal.resample_poly(ya,1,f,window = ('kaiser',2.6))
    zza = signal.resample_poly(za,1,f,window = ('kaiser',0.5))
    
    xxb = signal.resample_poly(xb,1,f,window = ('kaiser',1.5))
    yyb = signal.resample_poly(yb,1,f,window = ('kaiser',0.2))
    zzb = signal.resample_poly(zb,1,f,window = ('kaiser',0.0))

    #Se convierten los datos en centimetros *0.042
    #   (si se lee con EndoMIIDT) y posterior a milimetros
    xxa = xxa*1000
    yya = yya*1000
    zza = zza*1000
    
    xxb = xxb*1000
    yyb = yyb*1000
    zzb = zzb*1000


    #EndoViS Average Speed (mm/s)
    #Derecha
    SpeedD = np.sqrt(np.diff(xxa,1)**2 + np.diff(yya,1)**2 + np.diff(zza,1)**2)
    Mean_SpeedD = np.mean(SpeedD)
    
    #Izquierda
    SpeedI = np.sqrt(np.diff(xxb,1)**2 + np.diff(yyb,1)**2 + np.diff(zzb,1)**2)
    Mean_SpeedI = np.mean(SpeedI)
    #print("\nEndoViS Average Speed (mm/s): ", Mean_SpeedD)


    #EndoViS Average Acceleration (mm/s^2)
    #Derecha
    Accd = np.sqrt(np.diff(xxa,2)**2 + np.diff(yya,2)**2 + np.diff(zza,2)**2)
    Mean_AccD = np.mean(Accd)
    
    #Izquierda
    Acci = np.sqrt(np.diff(xxb,2)**2 + np.diff(yyb,2)**2 + np.diff(zzb,2)**2)
    Mean_AccI = np.mean(Acci)
    #print("\nEndoViS Average Acceleration (mm/s^2): ", Mean_AccD)


    #EndoViS Idle Time (%)
    #Derecha
    idle1D = np.argwhere(SpeedD<=5)
    idleD =(len(idle1D)/len(SpeedD))*100
    #print("\nEndoViS Idle Time (%): ", idleD)
    
    #Izquierda
    idle1I = np.argwhere(SpeedI<=5)
    idleI =(len(idle1I)/len(SpeedI))*100



    #EndoViS Max. Area (m^2)
    #Derecha
    max_horD = max(xa)-min(xa)
    max_vertD = max(ya)-min(ya)
    MaxAreaD = max_vertD*max_horD
    
    #Izquierda
    max_horI = max(xb)-min(xb)
    max_vertI = max(yb)-min(yb)
    MaxAreaI = max_vertI*max_horI

    #EndoViS Max. Volume (m^3)
    #Derecha
    max_altD = max(za)-min(za)
    MaxVolD = MaxAreaD*max_altD
    
    #Izquierda
    max_altI = max(zb)-min(zb)
    MaxVolI = MaxAreaI*max_altI

    #EndoViS Area/PL : EOA
    #Derecha
    A_PLD = np.sqrt(MaxAreaD)/PLD
    
    #Izquierda
    A_PLI = np.sqrt(MaxAreaI)/PLI
    #print("\nEndoViS Economy of Area (au.): ", A_PLD)

    #EndoViS Volume/PL: EOV
    #Derecha
    A_VD =  MaxVolD**(1/3)/PLD
    
    #Izquierda
    A_VI =  MaxVolI**(1/3)/PLI
    #print("\nEndoViS Economy of Volume (au.): ", A_VD)
    
    #EndoViS Bimanual Dexterity
    b= np.sum((SpeedI - Mean_SpeedI)*(SpeedD - Mean_SpeedD))
    d= np.sum(np.sqrt(((SpeedI - Mean_SpeedI)**2)*((SpeedD - Mean_SpeedD)**2)));   
    BD = b/d


    #EndoViS Energia
    #Derecha
    EXa = np.sum(xa**2)
    EYa = np.sum(ya**2)
    EZa = np.sum(za**2)

    EndoEAD = (EXa+EYa)/(MaxAreaD*100) #J/cm^2
    EndoEVD = (EXa+EYa+EZa)/(MaxVolD*100) #J/cm^3
    
    #Izquierda
    EXb = np.sum(xb**2)
    EYb = np.sum(yb**2)
    EZb = np.sum(zb**2)

    EndoEAI = (EXb+EYb)/(MaxAreaI*100) #J/cm^2
    EndoEVI = (EXb+EYb+EZb)/(MaxVolI*100) #J/cm^3
    
    #print("\nEndoViS Energy of Area (J/cm^2.): ", EndoEAD)
    #print("\nEndoViS Energy of Volume (J/cm^3.): ", EndoEVD)


    # Print parameters
    #EndoViS Tiempo
    print("\nEndoViS Tiempo (s): ", t)
    print("EndoViS Path Length (m.): ", PLD,PLI)
    print("EndoViS Depth Perception (m.): ", DPD,DPI)
    print("EndoViS Depth Perception along trocar", None)
    print('EndoViS Motion Smoothness 1: ', MS1,MS3)
    print('EndoViS Motion Smoothness 2: ', MS2,MS4)
    print("EndoViS Average Speed (mm/s): ", Mean_SpeedD, Mean_SpeedI)
    print("EndoViS Average Acceleration (mm/s^2): ", Mean_AccD,Mean_AccI)
    print("EndoViS Idle Time (%): ", idleD,idleI )
    print("EndoViS Economy of Area (au.): ", A_PLD,A_PLI)
    print("EndoViS Economy of Volume (au.): ", A_VD,A_VI)
    print("EndoViS Bimanual Dexterity", BD)
    print("EndoViS Energy of Area (J/cm^2.): ", EndoEAD, EndoEAI)
    print("EndoViS Energy of Volume (J/cm^3.): ", EndoEVD, EndoEVI)


    fo = open(ruta1 + "_derecha.csv", "a")
    fo.write(t1 + "," + str(PLD) + "," + str(DPD) + "," + "0" + "," + str(MS1) + "," + str(MS2).strip('[]') + "," + str(Mean_SpeedD) + "," + str(Mean_AccD) + "," + str(idleD) + "," + str(A_PLD) + "," + str(A_VD)  + "," + str(EndoEAD) + "," + str(EndoEVD) + "," + str(BD) + "\n")
    fo.close
    fo1 = open(ruta1 + "_izquierda.csv", "a")
    fo1.write(t1 + "," + str(PLI) + "," + str(DPI) + "," + "0" + "," + str(MS3) + "," + str(MS4).strip('[]') + "," + str(Mean_SpeedI) + "," + str(Mean_AccI) + "," + str(idleI) + "," + str(A_PLI) + "," + str(A_VI)  + "," + str(EndoEAI) + "," + str(EndoEVI) + "," + str(BD) + "\n")
    fo1.close
    
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
            
        if(max_area > 1000 and max_area< 2200 ):
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
    
        
    cx4 = 0
    cy4 = 0

    cx5 = 0
    cy5 = 0

    lastX4 = -1
    lastY4 = -1

    lastX5 = -1
    lastY5 = -1
    
    
    cx6 = 0
    cy6 = 0

    cx7 = 0
    cy7 = 0

    lastX6 = -1
    lastY6 = -1

    lastX7 = -1
    lastY7 = -1

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

        
        
        azules_bajos = np.array([45,60,6 ], dtype=np.uint8)
        azules_altos = np.array([183, 255, 255], dtype=np.uint8)


        # Iniciamos las cámaras

        cap = cv2.VideoCapture(0,cv2.CAP_V4L)


        cap.set(FRAME_PROP_WIDTH, frame_w)
        cap.set(FRAME_PROP_HEIGHT,frame_h)
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
            mask2 = cv2.inRange(hsv, azules_bajos, azules_altos)
            mask3 = cv2.inRange(hsv1, azules_bajos, azules_altos)
            mask4 = cv2.inRange(hsv2, verdes_bajos2, verdes_altos2)
            mask5 = cv2.inRange(hsv2, azules_bajos, azules_altos)
         
            
     
            
         

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
            
            eros2 = cv2.erode(mask2, kernel, iterations=2)
            maskOpen2 = cv2.morphologyEx(eros2, cv2.MORPH_OPEN, kernel)
            dila2 = cv2.dilate(maskOpen2,None,iterations=1)
            maskClose2 = cv2.morphologyEx(dila2, cv2.MORPH_CLOSE, kernel)
            maskClose2 = cv2.erode(maskClose2, kernel, iterations=2)
            
            eros3 = cv2.erode(mask3, kernel, iterations=2)
            maskOpen3 = cv2.morphologyEx(eros3, cv2.MORPH_OPEN, kernel)
            dila3 = cv2.dilate(maskOpen3,None,iterations=1)
            maskClose3 = cv2.morphologyEx(dila3, cv2.MORPH_CLOSE, kernel)
            maskClose3 = cv2.erode(maskClose3, kernel, iterations=2)
            
            eros4 = cv2.erode(mask4, kernel, iterations=1)
            maskOpen4 = cv2.morphologyEx(eros4, cv2.MORPH_OPEN, kernel)
            dila4 = cv2.dilate(maskOpen4,None,iterations=1)
            maskClose4 = cv2.morphologyEx(dila4, cv2.MORPH_CLOSE, kernel)
            maskClose4 = cv2.erode(maskClose4, kernel, iterations=1)
            
            eros5 = cv2.erode(mask5, kernel, iterations=2)
            maskOpen5 = cv2.morphologyEx(eros5, cv2.MORPH_OPEN, kernel)
            dila5 = cv2.dilate(maskOpen5,None,iterations=1)
            maskClose5 = cv2.morphologyEx(dila5, cv2.MORPH_CLOSE, kernel)
            maskClose5 = cv2.erode(maskClose5, kernel, iterations=2)
            
       

      
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

            maskFinal3 = maskClose3
            cnts3, h3 = cv2.findContours(
                maskFinal3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            maskFinal4 = maskClose4
            cnts4, h4 = cv2.findContours(
                maskFinal4, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            maskFinal5 = maskClose5
            cnts5, h5 = cv2.findContours(
                maskFinal5, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            

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

            max_area3 = 0
            for cnt3 in cnts3:
                area3 = cv2.contourArea(cnt3)
                if area3 > max_area3:
                    max_area3 = area3
                    cnt_best3 = cnt3
                    
            max_area4 = 0
            for cnt4 in cnts4:
                area4 = cv2.contourArea(cnt4)
                if area4 > max_area4:
                    max_area4 = area4
                    cnt_best4 = cnt4
                    
            max_area5 = 0
            for cnt5 in cnts5:
                area5 = cv2.contourArea(cnt5)
                if area5 > max_area5:
                    max_area5 = area5
                    cnt_best5 = cnt5
         
            #area1 = moments['m00']
            # área de la pantalla
            # print(area)
            if((max_area > 0  and max_area1 > 0) or (max_area2 > 0  and max_area3 > 0) or (max_area4 > 0  and max_area1 > 0) or (max_area5 > 0  and max_area3 > 0)):
                
                if (max_area > 0  and max_area1 > 0):
                    # Encontrar el area de los objetos que detecta la camara
                    moments = cv2.moments(cnt_best)
                    moments1 = cv2.moments(cnt_best1)
                

                    # Buscamos el centro x, y del objeto
                    cx = int(moments['m10']/moments['m00'])
                    cy = int(moments['m01']/moments['m00'])
                    cx1 = int(moments1['m10']/moments1['m00'])
                    cy1 = int(moments1['m01']/moments1['m00'])
                
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
                
              
                
                if (max_area2 > 0 and max_area3 > 0):
                    moments2 = cv2.moments(cnt_best2)
                    moments3 = cv2.moments(cnt_best3)
                    cx2 = int(moments2['m10']/moments2['m00'])
                    cy2 = int(moments2['m01']/moments2['m00'])
                    cx3 = int(moments3['m10']/moments3['m00'])
                    cy3 = int(moments3['m01']/moments3['m00'])
                    
                    #cv2.circle(imagen, (cx2, cy2), 7, (0, 0, 255), -1)
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
                
                if (max_area4 > 0 and max_area1 > 0):
                    moments4 = cv2.moments(cnt_best4)
                    moments5 = cv2.moments(cnt_best1)
                    cx4 = int(moments4['m10']/moments4['m00'])
                    cy4 = int(moments4['m01']/moments4['m00'])
                    cx5 = int(moments5['m10']/moments5['m00'])
                    cy5 = int(moments5['m01']/moments5['m00'])
                    
                    cv2.circle(imagen2, (cx4, cy4), 7, (255, 0, 0), -1)
                    
                   
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
                if (max_area5 > 0 and max_area3 > 0):
                    moments6 = cv2.moments(cnt_best5)
                    moments7 = cv2.moments(cnt_best3)
                    cx6 = int(moments6['m10']/moments6['m00'])
                    cy6 = int(moments6['m01']/moments6['m00'])
                    cx7 = int(moments7['m10']/moments7['m00'])
                    cy7 = int(moments7['m01']/moments7['m00'])
                    
                    cv2.circle(imagen2, (cx6, cy6), 7, (0, 0, 255), -1)
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
                   
                    
                
                
#                 cv2.putText(imagen,"Inicio",(coordenadas[0][0]-20,coordenadas[0][1]+42),font,0.5,(0, 0, 55),2)
#                 cv2.putText(imagen,"Fin",(coordenadas[-1][0]-10,coordenadas[-1][1]+42),font,0.5,(0, 0, 55),2)
               
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
                if ((lastX >= 0 and lastY >= 0 and cx >= 0 and cy >= 0) or (lastX2 >= 0 and lastY2 >= 0 and cx2 >= 0 and cy2 >= 0) or (lastX1 >= 0 and lastY1 >= 0 and cx1 >= 0 and cy1 >= 0) or (lastX3 >= 0 and lastY3 >= 0 and cx3 >= 0 and cy3 >= 0) or (lastX4 >= 0 and lastY4 >= 0 and cx4 >= 0 and cy4 >= 0) and (lastX6 >= 0 and lastY6 >= 0 and cx6 >= 0 and cy6 >= 0) or (lastX5 >= 0 and lastY5 >= 0 and cx5 >= 0 and cy5 >= 0) and (lastX7 >= 0 and lastY7 >= 0 and cx7 >= 0 and cy7 >= 0)):
                    # Dibuja recorrido del objeto
                    #cv2.line(imagen, (cx, cy), (lastX, lastY), (0, 255, 0), 2)
                    #cv2.line(imagen1, (cx1, cy1), (lastX1, lastY1), (0, 255, 0), 2)

                    if (cx4 ==coordenadas[0][0] and cy4==coordenadas[0][1]) or (cx4 ==coordenadas[0][0]+1 and cy4==coordenadas[0][1]) or (cx4 ==coordenadas[0][0]+2 and cy4==coordenadas[0][1]) or (cx4 ==coordenadas[0][0] and cy4==coordenadas[0][1]+1) or (cx4 ==coordenadas[0][0]+1 and cy4==coordenadas[0][1]+1) or (cx4 ==coordenadas[0][0]+2 and cy4==coordenadas[0][1]+1) or (cx4 ==coordenadas[0][0] and cy4==coordenadas[0][1]+2) or (cx4 ==coordenadas[0][0]+1 and cy4==coordenadas[0][1]+2) or (cx4 ==coordenadas[0][0]+2 and cy4==coordenadas[0][1]+2):
                    #if (cx4 ==coordenadas[1][0] and cy4==coordenadas[1][1]) or (cx4 ==coordenadas[1][0]+1 and cy4==coordenadas[1][1]) or (cx4 ==coordenadas[1][0]+2 and cy4==coordenadas[1][1]) or (cx4 ==coordenadas[1][0] and cy4==coordenadas[1][1]+1) or (cx4 ==coordenadas[1][0]+1 and cy4==coordenadas[1][1]+1) or (cx4 ==coordenadas[1][0]+2 and cy4==coordenadas[1][1]+1) or (cx4 ==coordenadas[1][0] and cy4==coordenadas[1][1]+2) or (cx4 ==coordenadas[1][0]+1 and cy4==coordenadas[1][1]+2) or (cx4 ==coordenadas[1][0]+2 and cy4==coordenadas[1][1]+2):
                    #if (cx4 ==coordenadas[2][0] and cy4==coordenadas[2][1]) or (cx4 ==coordenadas[2][0]+1 and cy4==coordenadas[2][1]) or (cx4 ==coordenadas[2][0]+2 and cy4==coordenadas[2][1]) or (cx4 ==coordenadas[2][0] and cy4==coordenadas[2][1]+1) or (cx4 ==coordenadas[2][0]+1 and cy4==coordenadas[2][1]+1) or (cx4 ==coordenadas[2][0]+2 and cy4==coordenadas[2][1]+1) or (cx4 ==coordenadas[2][0] and cy4==coordenadas[2][1]+2) or (cx4 ==coordenadas[2][0]+1 and cy4==coordenadas[2][1]+2) or (cx4 ==coordenadas[2][0]+2 and cy4==coordenadas[2][1]+2):
                   
                        cv2.circle(imagen2, (coordenadas[0][0],coordenadas[0][1]), 9, (0, 153, 255), 2)
                        inicio = 1
                        fin = 0
                        detener = 0

                        if inicio_tiempo == 0:
                            start_time = time()
                            inicio_tiempo = 1

                    if (cx6 ==coordenadas[-1][0] and cy6==coordenadas[-1][1]) or (cx6 ==coordenadas[-1][0]+1 and cy6==coordenadas[-1][1]) or (cx6 ==coordenadas[-1][0]+2 and cy6==coordenadas[-1][1]) or (cx6 ==coordenadas[-1][0] and cy6==coordenadas[-1][1]+1) or (cx6 ==coordenadas[-1][0]+1 and cy6==coordenadas[-1][1]+1) or (cx6 ==coordenadas[-1][0]+2 and cy6==coordenadas[-1][1]+1) or (cx6 ==coordenadas[-1][0] and cy6==coordenadas[-1][1]+2) or (cx6 ==coordenadas[-1][0]+1 and cy6==coordenadas[-1][1]+2) or (cx6 ==coordenadas[-1][0]+2 and cy6==coordenadas[-1][1]+2):
                    #if (cx6 ==coordenadas[-2][0] and cy6==coordenadas[-2][1]) or (cx6 ==coordenadas[-2][0]+1 and cy6==coordenadas[-2][1]) or (cx6 ==coordenadas[-2][0]+2 and cy6==coordenadas[-2][1]) or (cx6 ==coordenadas[-2][0] and cy6==coordenadas[-2][1]+1) or (cx6 ==coordenadas[-2][0]+1 and cy6==coordenadas[-2][1]+1) or (cx6 ==coordenadas[-2][0]+2 and cy6==coordenadas[-2][1]+1) or (cx6 ==coordenadas[-2][0] and cy6==coordenadas[-2][1]+2) or (cx6 ==coordenadas[-2][0]+1 and cy6==coordenadas[-2][1]+2) or (cx6 ==coordenadas[-2][0]+2 and cy6==coordenadas[-2][1]+2):
                    #if (cx6 ==coordenadas[-3][0] and cy6==coordenadas[-3][1]) or (cx6 ==coordenadas[-3][0]+1 and cy6==coordenadas[-3][1]) or (cx6 ==coordenadas[-3][0]+2 and cy6==coordenadas[-3][1]) or (cx6 ==coordenadas[-3][0] and cy6==coordenadas[-3][1]+1) or (cx6 ==coordenadas[-3][0]+1 and cy6==coordenadas[-3][1]+1) or (cx6 ==coordenadas[-3][0]+2 and cy6==coordenadas[-3][1]+1) or (cx6 ==coordenadas[-3][0] and cy6==coordenadas[-3][1]+2) or (cx6 ==coordenadas[-3][0]+1 and cy6==coordenadas[-3][1]+2) or (cx6 ==coordenadas[-3][0]+2 and cy6==coordenadas[-3][1]+2):                
                        cv2.circle(imagen2, (coordenadas[-1][0],coordenadas[-1][1]), 9, (0, 153, 255), 2)
                        inicio = 0
                        fin = 1

                    # Condición para guardar datos del recorrido y del tiempo
                    # Cuando se ha llegado al inicio pero falta llegar al final del recorrido,
                    # dependiendo si ya se han cumplido la cantidad de repeticiones

                    if inicio == 1 and fin == 0 and prueba <= int(pack.get()):

                        # Mostramos sus coordenadas por pantalla
                        print("c_m = ", cx,cy)
                        print("c_e= ", cx1,cy1)
                        print("c_m1 = ", cx2,cy2)
                        print("c_e1 = ", cx3,cy3)
                        print("c_m13 = ", cx4,cy4)
                        print("c_e13 = ", cx5,cy5)
                        print("c_m13 = ", cx6,cy6)
                        print("c_e13 = ", cx7,cy7)

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
                        x, y, z, x1, y1, z1 = conversion_2p(cx,cy,cx1,cy1,cx2,cy2,cx3,cy3)
                        
                        #Se almacenan las coordenadas pixel en un archivo csv
                        fo = open(e1.get() + '-Ejercicio-3-2' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" +
                                  "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                        fo.write(str(cx) + "," + str(cy) + "," + str(cy1) + "," + str(cx2) + "," + str(cy2) + "," + str(cy3) + "\n")
                        
                        #Se almacenan las coordenadas centimetro en un archivo csv
                        f1 = open(e1.get() + '-Ejercicio-3-2' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" +
                                  "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y") + "-cm.csv", "a")
                        f1.write(str(x) + "," + str(y) + "," + str(z) + "," + str(x1) + "," + str(y1) + "," + str(z1) + "\n")

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
# 
#                             cv2.putText(imagen1, "Fin de Repeticiones",
#                                         (35, 20), font, 0.6, (232, 189, 30), 2, 2)
#                             cv2.putText(imagen1, "Esc-Salir",
#                                         (230, 20), font, 0.6, (0, 255, 0), 2, 2)
                            
                            

                    # Si se finaliza el recorrido, se detiene el conteo de tiempo y almacena el tiempo final en TF
                    # A su vez, se guarda en el archivo de texto antes abierto en donde se guardaron las coordenadas de recorrido
                    if fin == 1:
                        if detener == 0:
                            ruta = e1.get() + '-Ejercicio-3-2' +"/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                            ruta0 = e1.get() + '-Ejercicio-3-2' +"/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")+"-cm"
                            ruta1 = e1.get() + '-Ejercicio-3-2' +"/" + "Maps" + "/" + e1.get() + "-Maps-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
                            ruta2 = e1.get() + '-Ejercicio-3-2' +"/" + "Graficas" + "/" + e1.get() + "-Grafica-" + "Rep" + str(prueba) + "-" + tiempo.strftime("%d-%m-%Y")
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
                            
                            
                            fo = open(e1.get() + '-Ejercicio-3-2' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" + "Rep" + str(
                                prueba) + "-" + tiempo.strftime("%d-%m-%Y") + ".csv", "a")
                            fo.write("TF" + "," + str(elapsed_time) + "," +""+ "," +""+ "," +""+"," +""+ "\n")
                            fo.close()
                            
                            f1 = open(e1.get() + '-Ejercicio-3-2' + "/" + "Repeticiones" + "/" + e1.get() + "-Camaras-" + "Rep" + str(
                                prueba) + "-" + tiempo.strftime("%d-%m-%Y") + "-cm.csv", "a")
                            f1.write("TF" + "," + str(elapsed_time) + "," +""+ "," +""+ "," +""+"," +""+ "\n")
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
                lastX4, lastY4 = cx4, cy4
                lastX5, lastY5 = cx5, cy5
                lastX6, lastY6 = cx6, cy6
                lastX7, lastY7 = cx7, cy7

                # Mostramos la imagen original con la marca del centro y
                # la mascara

                #cv2.imshow('Mascara1', maskClose1)
                #cv2.imshow('brillo', hsv)
                #cv2.imshow('Difuminado', frame)
                #cv2.imshow('Mascara2', maskClose)
                #cv2.imshow('Mascara3', maskClose4)
                cv2.imshow('Laparoscopia: EJercicio 3-2', imagen2)
                #cv2.imshow('Camara1_Maestra', imagen)
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

Label(master, text="Traslado físico de objetos \ncon 2 pinzas", bg='#ededed').grid(
    row=0, column=1, columnspan=2)

Label(master, text="Usuario:", bg='#ededed').grid(row=2, column=0)
e1 = Entry(master, width=24, bg='#FFFFFF')
e1.grid(row=2, column=1)

#Label(master, text="Tarea:", bg='#ededed').grid(row=3, column=0)

#fig = ttk.Combobox(master, state="readonly", values=('Transferencia'))
#fig.grid(row=3, column=1, columnspan=2, sticky=W, pady=4)
#fig.set("Transferencia")

Label(master, text="Repeticiones:", bg='#ededed').grid(row=3, column=0)

pack = ttk.Combobox(master, state="readonly", values=('3', '4', '5'))
pack.grid(row=3, column=1, columnspan=2, sticky=W, pady=4)
pack.set("3")

Button(master, text='Iniciar', cursor="hand2", fg='Black', bg='#f7c282', width=20, command=lambda: createDir(
    e1.get() + '-Ejercicio-3-2')).grid(row=4, column=1, columnspan=2, sticky=W, pady=4)
Button(master, text='Analizar', cursor="hand2", fg='Black', bg='#31b6fd', width=20, command=abrir).grid(row=5, column=1, columnspan=2, sticky=W, pady=4)

Button(master, text='Salir', cursor="hand2", fg='White', bg='#bb2003',
       width=20, command=master.destroy).grid(row=6, column=1, columnspan=2, sticky=W, pady=4)

master.mainloop()
