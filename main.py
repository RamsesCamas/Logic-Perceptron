import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, Entry, font

def step(u):
    return np.heaviside(u,1)

def perceptron(X,W):
    u = X.dot(W)
    return step(u)

def train(X,W, alpha, yd):
    errors = []
    yc = perceptron(X,W)
    ek = yd - yc
    errors.append(ek.sum())
    Wk = W
    i = 0
    while ek.sum() != 0:
      Wk = Wk  + (alpha*(ek.T.dot(X)))
      yc = perceptron(X,Wk)
      ek = yd - yc
      errors.append(ek.sum())
      i+=1
    print(f'Entrenado en {i} iteraciones')
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

def execute_nn():
    X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    W0 = np.random.rand(3)
    label_FW['text'] = f'W0: {W0}'
    A1 = float(entry_alpha1.get())
    A2 = float(entry_alpha2.get())
    A3 = float(entry_alpha3.get())
    A4 = float(entry_alpha4.get())
    A5 = float(entry_alpha5.get())
    alphas = [A1,A2,A3,A4,A5] 

    yd0 = int(entry_Y0.get())
    yd1 = int(entry_Y1.get())
    yd2 = int(entry_Y2.get())
    yd3 = int(entry_Y3.get())
    yd = np.array([yd0,yd1,yd2,yd3]) 

    total_errors, W_s = [] , []
    for alpha in alphas:
        error, F_W = train(X,W0,alpha,yd) 
        total_errors.append(error)
        W_s.append(F_W)
    max_total_errors = len(max(total_errors,key=len))
    Y_s = normalize_arrays(total_errors,max_total_errors)

    get_and = yd == np.array([0,0,0,1])
    get_or = yd == np.array([0,1,1,1])
    title = ''
    if(get_and.all()): title = 'AND'
    if(get_or.all()): title = 'OR'

    x = np.arange(0,len(Y_s[0]))
    my_colors = ['r','g','b','c','m']
    label_Ws = [label_W0,label_W1,label_W2,label_W3,label_W4]
    for y, my_color, alpha,my_label, my_W in zip(Y_s,my_colors,alphas,label_Ws,W_s):
        plt.plot(x,y,color=my_color,label=f'Alpha {alpha}')
        my_label['text'] = f'W final con Alpha {alpha}: {my_W}'
    plt.legend(loc="lower right")
    plt.xlabel('Iteraciones')
    plt.ylabel('Error')
    plt.title(title)
    plt.show()

if __name__=='__main__':
    root = Tk()
    my_font = font.Font(size=13)
    root.geometry('800x400')
    root.title('Perceptrón')

    label_alpha1 = Label(root, width=15, font=my_font,text='Alpha 1')
    label_alpha1.place(x=30,y=20)
    label_alpha2 = Label(root, width=15, font=my_font,text='Alpha 2')
    label_alpha2.place(x=200,y=20)
    label_alpha3 = Label(root, width=15, font=my_font,text='Alpha 3')
    label_alpha3.place(x=350,y=20)
    label_alpha4 = Label(root, width=15, font=my_font,text='Alpha 4')
    label_alpha4.place(x=500,y=20)
    label_alpha5 = Label(root, width=15, font=my_font,text='Alpha 5')
    label_alpha5.place(x=650,y=20)

    entry_alpha1 = Entry(root,width=14, font=my_font)
    entry_alpha1.place(x=30,y=50)
    entry_alpha2 = Entry(root,width=14, font=my_font)
    entry_alpha2.place(x=200,y=50)
    entry_alpha3 = Entry(root,width=14, font=my_font)
    entry_alpha3.place(x=350,y=50)
    entry_alpha4 = Entry(root,width=14, font=my_font)
    entry_alpha4.place(x=500,y=50)
    entry_alpha5 = Entry(root,width=14, font=my_font)
    entry_alpha5.place(x=650,y=50)

    label_Ys = Label(root, width=15, font=my_font, text='Y determinada:')
    label_Ys.place(x=20,y=150)

    entry_Y0 = Entry(root,width=8, font=my_font)
    entry_Y0.place(x=170,y=150)
    entry_Y1 = Entry(root,width=8, font=my_font)
    entry_Y1.place(x=270,y=150)
    entry_Y2 = Entry(root,width=8, font=my_font)
    entry_Y2.place(x=370,y=150)
    entry_Y3 = Entry(root,width=8, font=my_font)
    entry_Y3.place(x=470,y=150)

    btn_exec_nn = Button(root, width=14, text='Ejecutar', font=my_font, command=execute_nn)
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

    label_and = Label(root, width=15, font=my_font, text='Y AND: 0 0 0 1')
    label_and.place(x=600,y=220)
    label_or = Label(root, width=15, font=my_font, text='Y OR: 0 1 1 1')
    label_or.place(x=600,y=250)

    root.mainloop()