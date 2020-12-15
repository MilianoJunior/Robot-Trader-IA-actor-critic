# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:17:11 2020

@author: Administrador
"""
import numpy as np
import tensorflow as tf
from comunica import  Comunica
print(tf.__version__)

media = np.array([ 1.34059340e+03, -7.24600000e-01,  1.72143000e+01,  1.27376000e+01,
        4.93518260e+01, -4.01833500e+00, -8.25555400e+00,  4.23728500e+00,
        2.93417980e+01,  5.81721490e+01, -1.71219369e+05])

std = np.array([2.66276314e+02, 3.87526532e+01, 1.50242786e+01, 1.30033017e+01,
       1.23548227e+01, 5.51075722e+01, 9.42626581e+01, 4.15846854e+01,
       1.10544609e+01, 1.81661716e+01, 1.16718646e+05])

colunas = ['Hora','dif', 'retracao +','retracao -', 'RSI'] #, 'M22M44', 'M22M66', 'M66M44', 'ADX', 'ATR','OBV']
new_model = tf.keras.models.load_model('modelo_02')
# trader1 =tf.keras.models.load_model('IA_ESPECIALISTA11Pontos_5.h5')
HOST = ''    # Host
PORT = 8888  # Porta
R = Comunica(HOST,PORT)
s = R.createServer()

while True:
    p,addr = R.runServer(s)
    jm = (p-media)/std
    jm = np.array([jm])
    state = tf.constant(jm, dtype=tf.float32)
    previsao1 = new_model.predict(state)
    # print(jm)
    # print(p)
    print('   ')
    print(previsao1)
    print('   ')
    previsao2 = np.argmax(previsao1[0][0])
    d3 = previsao1[0][0][previsao2]
    if previsao2 == 0:
        print('Sem operacao')
    if previsao2 == 1:
        flag ="compra-{}".format(d3)
        print('compra: ',previsao2)
        R.enviaDados(flag,s,addr)
    if previsao2 == 2:
        flag = "venda-{}".format(d3)
        print('venda: ',previsao2)
        R.enviaDados(flag,s,addr)