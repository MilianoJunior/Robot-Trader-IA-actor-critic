# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:54:27 2020

@author: jrmfi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from comunica import  Comunica


media = np.array([ 1.33407966e+03, -6.91781964e-03,  2.29993332e+01,  2.11906984e+01,
        5.03690652e+01,  3.00455726e+00,  6.00056243e+00, -2.99601117e+00,
        2.99655274e+01,  8.32517268e+01,  1.00003863e+02, -9.13884809e+02])

std = np.array([2.61128920e+02, 5.30899357e+01, 2.03602922e+01, 1.99119561e+01,
       1.24189700e+01, 8.43612992e+01, 1.45021113e+02, 6.43614859e+01,
       1.17763257e+01, 3.04029743e+01, 2.35761966e-01, 2.01115685e+05])

new_model = tf.keras.models.load_model('modelo_16')

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


