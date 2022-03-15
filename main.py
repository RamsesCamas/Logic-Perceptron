import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, Entry, font

def step(u):
    return np.heaviside(u,1)

def perceptron(X,W):
    u = X.dot(W)
    return step(u)

def entrenar(X,W, taza_aprendizaje, yd):
    errors = []
    norm_err = 1
    Wk = W
    while norm_err != 0:
        yc = perceptron(X,Wk)
        ek = yd - yc
        norm_err = np.linalg.norm(ek)
        Wk = Wk  + (taza_aprendizaje*(ek.T.dot(X)))
        errors.append(norm_err)
    return errors , Wk

def normalize_arrays(total_errors, max_total_errors):
    Y_s = []
    for err in total_errors:
        size_list = len(err)
        error_array = np.array(err)
        size_diff = max_total_errors - size_list
        if size_diff > 0:
            rest_of_array = np.zeros(size_diff, dtype=int)
            error_array = np.concatenate((error_array, rest_of_array))
        Y_s.append(error_array)
    return Y_s

def ejecutar_perceptron():
    X_dos_entradas = np.array([[1,0,0],
                               [1,0,1],
                               [1,1,0],
                               [1,1,1]])

    W0_dos_entradas = np.random.rand(3)
    X_tres_entradas = np.array([[1,0,0,0],
                                [1,0,0,1],
                                [1,0,1,0],
                                [1,0,1,1],
                                [1,1,0,0],
                                [1,1,0,1],
                                [1,1,1,0],
                                [1,1,1,1]])

    W0_tres_entradas = np.random.rand(4)
    label_FW['text'] = f'Pesos iniciales: {W0_dos_entradas}'
    TA1 = float(entry_taza_aprendizaje1.get())
    TA2 = float(entry_taza_aprendizaje2.get())
    TA3 = float(entry_taza_aprendizaje3.get())
    TA4 = float(entry_taza_aprendizaje4.get())
    TA5 = float(entry_taza_aprendizaje5.get())
    taza_aprendizajes = [TA1,TA2,TA3,TA4,TA5] 

    Y = str(entry_yd.get())
    Yd = np.array(list(map(int, Y.split(','))))

    total_errors, W_s = [] , []
    for taza_aprendizaje in taza_aprendizajes:
        error, F_W = entrenar(X_dos_entradas,W0_dos_entradas,taza_aprendizaje,Yd) 
        total_errors.append(error)
        W_s.append(F_W)
    max_total_errors = len(max(total_errors,key=len))
    Y_s = normalize_arrays(total_errors,max_total_errors)

    title = 'Norma del error'

    x = np.arange(0,len(Y_s[0]))

    plt.plot(x,Y_s[0],color='green',label=f'taza_aprendizaje {TA1}')
    label_W0['text'] = f'Pesos finales con taza_aprendizaje {TA1}: {W_s[0]}'
    plt.plot(x,Y_s[1],color='red',label=f'taza_aprendizaje {TA2}')
    label_W1['text'] = f'Pesos finales con taza_aprendizaje {TA1}: {W_s[1]}'
    plt.plot(x,Y_s[2],color='blue',label=f'taza_aprendizaje {TA3}')
    label_W2['text'] = f'Pesos finales con taza_aprendizaje {TA1}: {W_s[2]}'
    plt.plot(x,Y_s[3],color='orange',label=f'taza_aprendizaje {TA4}')
    label_W3['text'] = f'Pesos finales con taza_aprendizaje {TA1}: {W_s[3]}'
    plt.plot(x,Y_s[4],color='cyan',label=f'taza_aprendizaje {TA5}')
    label_W4['text'] = f'Pesos finales con taza_aprendizaje {TA1}: {W_s[4]}'
    
    plt.legend(loc="upper right")
    plt.xlabel('Iteraciones')
    plt.ylabel('Error')
    plt.title(title)
    plt.show()

if __name__=='__main__':
    root = Tk()
    my_font = font.Font(size=13)
    root.geometry('800x400')
    root.title('Perceptrón')

    label_taza_aprendizaje1 = Label(root, width=15, font=my_font,text='taza_aprendizaje 1')
    label_taza_aprendizaje1.place(x=30,y=20)
    label_taza_aprendizaje2 = Label(root, width=15, font=my_font,text='taza_aprendizaje 2')
    label_taza_aprendizaje2.place(x=200,y=20)
    label_taza_aprendizaje3 = Label(root, width=15, font=my_font,text='taza_aprendizaje 3')
    label_taza_aprendizaje3.place(x=350,y=20)
    label_taza_aprendizaje4 = Label(root, width=15, font=my_font,text='taza_aprendizaje 4')
    label_taza_aprendizaje4.place(x=500,y=20)
    label_taza_aprendizaje5 = Label(root, width=15, font=my_font,text='taza_aprendizaje 5')
    label_taza_aprendizaje5.place(x=650,y=20)

    entry_taza_aprendizaje1 = Entry(root,width=14, font=my_font)
    entry_taza_aprendizaje1.place(x=30,y=50)
    entry_taza_aprendizaje2 = Entry(root,width=14, font=my_font)
    entry_taza_aprendizaje2.place(x=200,y=50)
    entry_taza_aprendizaje3 = Entry(root,width=14, font=my_font)
    entry_taza_aprendizaje3.place(x=350,y=50)
    entry_taza_aprendizaje4 = Entry(root,width=14, font=my_font)
    entry_taza_aprendizaje4.place(x=500,y=50)
    entry_taza_aprendizaje5 = Entry(root,width=14, font=my_font)
    entry_taza_aprendizaje5.place(x=650,y=50)

    label_Ys = Label(root, width=15, font=my_font, text='Y determinada:')
    label_Ys.place(x=20,y=150)

    entry_yd = Entry(root,width=50,font=my_font)
    entry_yd.place(x=350,y=200)

    label_yd = Label(root,width=30,font=my_font,text='Y determinada').place(x=350,y=170)

    btn_exec_nn = Button(root, width=14, text='Ejecutar', font=my_font, command=ejecutar_perceptron)
    btn_exec_nn.place(x=600,y=145)

    label_FW = Label(root, width=70, font=my_font, text='')
    label_FW.place(x=0,y=100)
    label_W0 = Label(root, width=70, font=my_font, text='')
    label_W0.place(x=0,y=220)
    label_W1 = Label(root, width=70, font=my_font, text='')
    label_W1.place(x=0,y=250)
    label_W2 = Label(root, width=70, font=my_font, text='')
    label_W2.place(x=0,y=280)
    label_W3 = Label(root, width=70, font=my_font, text='')
    label_W3.place(x=0,y=310)
    label_W4 = Label(root, width=70, font=my_font, text='')
    label_W4.place(x=0,y=340)


    root.mainloop()