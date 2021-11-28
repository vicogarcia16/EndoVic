from tkinter import *
from tkinter import messagebox
import subprocess
import os
import sys
   
def center(ventana):
    ventana.update_idletasks()
    w=ventana.winfo_width()
    h=ventana.winfo_height()
    extraW=ventana.winfo_screenwidth()-w
    extraH=ventana.winfo_screenheight()-h
    ventana.geometry("%dx%d%+d%+d" % (w,h,extraW/2,extraH/2))

def ejecutar(archivo):
    if sys.platform == "win32":
        os.system (archivo)
    else:
        os.system ("/usr/bin/python3 " + archivo)
        #os.system ('/User' + archivo)

def ventana2():
    master = Tk()
    master.title("Cirugía Laparoscópica")
    master.resizable(0,0)
    if sys.platform == "win32":
        master.geometry("400x220")
    else:
        master.geometry("393x220")
        master.config(bg="#ededed")

    center(master)


    Label(master,text="Traslado físico de objetos", bg="#ededed", height=4).grid(row=0,column=0,columnspan=4)
    Button(master, text='Ejercicio con \n1 pinza', relief="groove", cursor="hand2", fg='Black', bg='#f7c282', width=11, height=5, command=lambda: ejecutar("ejercicio3_1pinza_3cam.py"),activebackground="#31b6fd").grid(row=4, column=1, sticky=W, padx = 2, pady=3)
    Button(master, text='Ejercicio con \n2 pinzas', relief="groove", cursor="hand2", fg='Black', bg='#f7c282', width=11, height=5, command=lambda: ejecutar("ejercicio3_2pinzas_3cam.py"),activebackground="#31b6fd").grid(row=4, column=2, sticky=W, padx = 2, pady=3)         
    Button(master, text='Salir', relief="groove", cursor="hand2", fg='White', bg='#bb2003', width=11, height=5, command=master.destroy).grid(row=4, column=3, sticky=W, padx= 2, pady= 3)

    master.mainloop()


master = Tk()
master.title("Cirugía Laparoscópica")
master.resizable(0,0)
if sys.platform == "win32":
    master.geometry("475x220")
else:
    master.geometry("523x220")
    master.config(bg="#ededed")

center(master)


Label(master,text="Sistema para la clasificación objetiva de habilidades", bg="#ededed", height=4).grid(row=0,column=1,columnspan=4)
Button(master, text='Ejercicio 1', relief="groove", cursor="hand2", fg='Black', bg='#f7c282', width=11, height=5, command=lambda: ejecutar("ejercicio1_1pinza_3cam.py"),activebackground="#31b6fd").grid(row=4, column=1, sticky=W, padx = 2, pady=3)
Button(master, text='Ejercicio 2', relief="groove", cursor="hand2", fg='Black', bg='#f7c282', width=11, height=5, command=lambda: ejecutar("ejercicio2_3cam.py"),activebackground="#31b6fd").grid(row=4, column=2, sticky=W, padx = 2, pady=3) 
Button(master, text='Ejercicio 3', relief="groove", cursor="hand2", fg='Black', bg='#f7c282', width=11, height=5, command=lambda: ventana2(),activebackground="#31b6fd").grid(row=4, column=3, sticky=W, padx = 2, pady=3)        
Button(master, text='Salir', relief="groove", cursor="hand2", fg='White', bg='#bb2003', width=11, height=5, command=master.destroy).grid(row=4, column=4, sticky=W, padx= 2, pady= 3)

master.mainloop()