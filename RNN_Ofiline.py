# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 07:10:11 2020

@author: Administrador
"""
import numpy as np
import tensorflow as tf
from comunica import  Comunica
print(tf.__version__)

with tf.device('/CPU:0'):
    media = np.array([ 1.34240541e+03,  1.31531532e+00,  2.30990991e+01,  2.05945946e+01,
            5.07510450e+01,  2.37560360e+00,  5.67790991e+00, -3.30214414e+00,
            2.93371712e+01,  8.54033874e+01,  9.99992793e+01,  7.98845245e+03,
            6.22600721e+03,  3.16659459e+00, -4.22933333e+01,  4.14183784e+01,
            4.94940901e+01,  4.95253333e+01, -5.11105586e+01,  9.42991892e+01,
            5.14390270e+01,  1.14591303e+05,  1.14779901e+05,  1.14402705e+05])
    
    std = np.array([2.66985761e+02, 5.30349626e+01, 1.87958609e+01, 1.64137492e+01,
           1.15304199e+01, 6.63198330e+01, 1.11233217e+02, 4.80964716e+01,
           1.01216885e+01, 1.99421339e+01, 1.94173764e-01, 1.39156450e+05,
           3.71850523e+03, 1.05921225e+02, 9.73567455e+01, 9.78732289e+01,
           2.43338699e+01, 2.25903523e+01, 2.87026075e+01, 6.10559563e+01,
           1.77633441e+01, 3.59000302e+02, 3.88236638e+02, 3.69942603e+02])
    
    # '2020.12.11'
    a1 = np.array([900.0, 110, 50, 0, 25.78, 11.59, 31.23, -19.64, 19.74, 138.57,
           99.47, -423311.54, 8735.0, -373.66, -736.03, -576.03, 35.54, 66.05,
           -89.0, 171.76, 47.16, 115094.25, 115437.78, 114750.72],dtype=object)
    a2 = np.array([904.0, -10, 50, 15, 23.8, -88.72, -107.58, 18.86, 14.84, 161.79,
           99.35, -269015.38, 4996.0, -108.26, -453.24, -378.24, 8.06, 7.43,
           -94.29, 333.91, 43.17, 114963.25, 115631.06, 114295.44],dtype=object)
    a3 = np.array([908.0, -110, 15, 20, 27.98, -146.78, -195.38, 48.6, 11.1, 172.14,
           99.19, -457021.15, 6423.0, -67.87, -284.51, -139.51, 37.5, 35.27,
           -96.76, 402.58, 46.61, 114818.75, 115623.91, 114013.59],
          dtype=object)
    
    def verificacao(p,a1):
        
        if a1[0] == p[0]:
            print(p)
            print('Horario Igual: ',a1[0])
            for i in range(len(p)-1):
                if a1[i] == p[i]:
                    print(i,'ok')
                else:
                    print(i,'-',a1[i],'-',p[i],' dif: ',a1[i]-p[i])
        else:
            return 0
    # [908.0, -35.0, 5.0, 10.0, 72.82, 114.96, 162.58, -47.62, 35.41, 83.93, 100.4, 29.23, -2303847.0, 0.0, 38.83, 75.81, 75.81, 60.87, 55.17, -24.07, 215.47, -815209.38, 48.49, 100556.75, 100987.69, 100125.81]
    new_model = tf.keras.models.load_model('modelo_14')
    # trader1 =tf.keras.models.load_model('IA_ESPECIALISTA11Pontos_5.h5')
    HOST = ''    # Host
    PORT = 8888  # Porta
    R = Comunica(HOST,PORT)
    s = R.createServer()
    
    while True:
        p,addr = R.runServer(s)
        verificacao(p,a1)
        verificacao(p,a2)
        verificacao(p,a3)
        jm = (p-media)/std
        jm = np.array([jm])
        state = tf.constant(jm, dtype=tf.float32)
        previsao1 = new_model.predict(state)
        previsao2 = np.argmax(previsao1[0][0])
        d3 = p[0]
        print('recebido: ',p[0])
        # print(previsao2)
        # print('----------------')
        # d3 = previsao1[0][0][previsao2]
        if previsao2 == 0:
            print('Sem operacao')
        if previsao2 == 1:
            flag = "compra-{}".format(d3)
            # flag ="compra"
            print('compra: ',previsao2)
            R.enviaDados(flag,s,addr)
        if previsao2 == 2:
            flag = "venda-{}".format(d3)
            # flag = "venda"
            print('venda: ',previsao2)
            R.enviaDados(flag,s,addr)