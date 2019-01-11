import math
import numpy as np
from tkinter import *
import sys

root = Tk()
w11_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
w12_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
w13_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
w21_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
w22_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
w23_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
w31_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
w32_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
wb1_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
wb2_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
wb3_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
y_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)
y1_list = Listbox(root,width = 10, height = 15, selectmode = BROWSE)

w11_label = Label(root,text = "w11")
w12_label = Label(root,text = "w12")
w13_label = Label(root,text = "w13")
w21_label = Label(root,text = "w21")
w22_label = Label(root,text = "w22")
w23_label = Label(root,text = "w23")
w31_label = Label(root,text = "w31")
w32_label = Label(root,text = "w32")
wb1_label = Label(root,text = "wb1")
wb2_label = Label(root,text = "wb2")
wb3_label = Label(root,text = "wb3")
y_label = Label(root,text = "y")
y1_label = Label(root,text = "o")

giris = Entry(root, width="8")
giris.pack(side=LEFT)


w11 = -2.11
w12 = 0.69
w21 = 1.83
w22 = 1.12
w31 = 1.49
w32 = 1.97
w13 = -2.89
w23 = -1.36
wb1 = -0.24
wb2 = -2.4
wb3 = -2.12
wb = 1

delta_w11 = 0
delta_w12 = 0
delta_w13 = 0
delta_w21 = 0
delta_w22 = 0
delta_w23 = 0
delta_w31 = 0
delta_w32 = 0
delta_wb1 = 0
delta_wb2 = 0
delta_wb3 = 0
delta_wb = 0

f1 = 0
f2 = 0
f3 = 0

cikis_hatasi = 0
h_p = 0.5
geriye_yayilma_hatasi = 0
alfa = 1

s1 = 0
s2 = 0

x1_input = [0,0,0,0,1,1,1,1]
x2_input = [0,0,1,1,0,0,1,1]
x3_input = [0,1,0,1,0,1,0,1]

y_output = [0,1,0,0,0,0,1,1]

for j in range(10000):

    for i in range(8):
        net1 = (x1_input[i]*w11) + (x2_input[i]*w21) + (x3_input[i] + wb1)
        f1 = 1 / (1 + math.exp(net1*-1))
        #print('F1 : ' , f1)

        net2 = (x1_input[i]*w12) + (x2_input[i]*w22) + (x3_input[i] + wb2)
        f2 = 1 / (1 + math.exp(net2*-1))
        #print('F2 :', f2)

        net3 = (f1*w13) + (f2*w23) + (wb*wb3)
        f3 = 1 / (1 + math.exp(net3*-1))
        #print('F3: ', f3)

        
        a = x1_input[i]
        
        b = x2_input[i]
        
        c = x3_input[i]

        input_data = np.array([a, b, c])

        weights = {
            'node_0': np.array([w11, w21, w31]),
            'node_1': np.array([w12, w22, w32]),
            'node_2': np.array([w13, w23, 0]),
            'output_node': np.array([3])
        }

        
        node_0_value = (input_data * weights['node_0']).sum()
        
        #print('node 0_hidden: {}'.format(node_0_value))

        node_1_value = (input_data * weights['node_1']).sum()
        
        #print('node_1_hidden: {}'.format(node_1_value))

        node_2_value = (input_data * weights['node_2']).sum()
        
        #print('node_2_hidden: {}'.format(node_1_value))

        hidden_layer_values = np.array([node_0_value, node_1_value, node_2_value])

        output_layer = (hidden_layer_values * weights['output_node']).sum()

        #print("output layer : {}".format(output_layer))

        cikis_hatasi = h_p*(y_output[i] - f3)**2
        #print('Cikis Hatasi: ' , cikis_hatasi)
        #print(f3)

        if(cikis_hatasi != 0):
            geriye_yayilma_hatasi = f3*(1-f3)*(y_output[i]-f3)
            #print('Geriye Yayılma Hatasi : ' ,geriye_yayilma_hatasi)
            #print('Yeni Agirlik Degerleri Hesaplanacak')

            delta_wb3 = alfa*(geriye_yayilma_hatasi)*(wb)
            wb3 = wb3 + delta_wb3
            #print('Yeni Wb3 : ', wb3)

            delta_w13 = alfa*(geriye_yayilma_hatasi)*f1
            w13 = w13 + delta_w13
            #print('Yeni W13 : ', w13)

            delta_w23 = alfa*(geriye_yayilma_hatasi)*f2
            w23 = w23 + delta_w23
            #print('Yeni W23 : ', w23)

            #print('Gizli Katman icin Agirlik Hesaplama')
            s1 = f1*(1-f1)*(w13)*(geriye_yayilma_hatasi)
            #print(s1)
            wb1 = wb1 + s1

            delta_w11 = alfa*(s1)*x1_input[i]
            w11 = w11 + delta_w11
            #print(w11)

            delta_w21 = alfa*(s1)*x2_input[i]
            w21 = w21 + delta_w21
            #print('w21 : ' , w21)

            delta_w31 = alfa*(s1)*x3_input[i]
            w31 = w31 + delta_w31
            #print('W31 : ', w31)

            s2 = f2*(1-f2)*w23*(geriye_yayilma_hatasi)
            #print('s2 = ', s2)

            delta_wb2 = alfa*(s2)*wb
            wb2 = wb2 + delta_wb2
            #print(wb2)

            delta_w12 = alfa*(s2)*x1_input[i]
            w12 = w12 + delta_w12
            #print(w12)

            delta_w22 = alfa*(s2)*x2_input[i]
            w22 = w22 + delta_w22
            #print(w22)

            delta_w32 = alfa*(s2)*x3_input[i]
            w32 = w32 + delta_w32
            #print(w32)
            #print('w11: ', w11)
            #print('w12: ', w12)
            #print('w21: ', w21)
            #print('w22: ', w22)
            #print('w31: ', w31)
            #print('w32: ', w32)
            #print('w13: ', w13)
            #print('w23: ', w23)
            #print('wb1: ', wb1)
            #print('wb2: ', wb2)
            #print('wb3: ', wb3)
            w11_list.insert(i, w11)
            w12_list.insert(i, w12)
            w13_list.insert(i, w13)
            w21_list.insert(i, w21)
            w22_list.insert(i, w22)
            w23_list.insert(i, w23)
            w31_list.insert(i, w31)
            w32_list.insert(i, w32)
            wb1_list.insert(i, wb1)
            wb2_list.insert(i, wb2)
            wb3_list.insert(i, wb3)
            y_list.insert(i, y_output[i])
            y1_list.insert(i, output_layer)
            
w11_label.pack(side=LEFT)
w11_list.pack(side = LEFT)
w12_label.pack(side=LEFT)
w12_list.pack(side = LEFT)
w13_label.pack(side=LEFT)
w13_list.pack(side = LEFT)
w21_label.pack(side=LEFT)
w21_list.pack(side = LEFT)
w22_label.pack(side=LEFT)
w22_list.pack(side = LEFT)
w23_label.pack(side=LEFT)
w23_list.pack(side = LEFT)
w31_label.pack(side=LEFT)
w31_list.pack(side = LEFT)
w32_label.pack(side=LEFT)
w32_list.pack(side = LEFT)
wb1_label.pack(side=LEFT)
wb1_list.pack(side = LEFT)
wb2_label.pack(side=LEFT)
wb2_list.pack(side = LEFT)
wb3_label.pack(side=LEFT)
wb3_list.pack(side = LEFT)
y_label.pack(side=LEFT)
y_list.pack(side = LEFT)
y1_label.pack(side=LEFT)
y1_list.pack(side = LEFT)



root.geometry("1600x1600+120+120")

root.mainloop()  
        

        
