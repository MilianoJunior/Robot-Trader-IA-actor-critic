# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:19:14 2020

@author: Administrador
"""
import pandas as pd

class Trade():
    comprado = False
    vendido = False
    ficha = 0
    cont = 0
    valor = 0
    contC = 0
    contV = 0
    hora = 0
    Stop = 100
    posicao = 0

    def __init__(self,name='wind'):
        self.name = name
        self.CC = []
        self.VV = []
        self.saida = pd.DataFrame(columns=['tipo','entrada','saida','ganho','inicio','fim','duracao','ganhofinal','acumulado','recompensas'])
        self.rewards = []
        
        
    def agente(self,dados,acao,stop,gain,verbose=0):
        self.cont += 1    
        self.posicao = 0
        if acao == 0:
            p = 0
            # self.posicao = self.verifica(dados,gain,stop,verbose)
            # return self.CC,self.VV, self.saida,self.ficha,self.comprado,self.vendido,self.valor
        elif acao == 1 and self.comprado == False:
            if self.ficha == 0:
                self.ficha = 1
                self.comprado = True
                self.vendido = False
            else:
                self.ficha = 0
                self.comprado = False
                self.vendido = False
            # self.rewards.append(dados[6])
            self.posicao = self.compra(dados[1],dados[0],self.ficha,verbose)
            # self.verifica(dados,gain,stop,verbose)
            
        elif acao == 2 and self.vendido == False: 
            if self.ficha == 0:
                self.ficha = 1
                self.vendido = True
                self.comprado = False
            else:
                self.ficha = 0
                self.vendido = False
                self.comprado = False
            # self.rewards.append(dados[6])
            self.posicao = self.venda(dados[1],dados[0],self.ficha,verbose)
            # self.verifica(dados,gain,stop,verbose)
            
        if self.ficha == 1:
            rec = self.valor -dados[4]
            # self.posicao = self.verifica(dados,gain,stop,verbose)
            # self.imprimi(dados)
            if self.vendido:
                self.rewards.append(rec)
            else:
                self.rewards.append(-rec)
        if self.comprado or self.vendido:
            self.posicao = self.verifica(dados,gain,stop,verbose)
        if verbose == 1:
            print('##########################')
            print('Contador: ',self.cont)
            print('Posicao: ',self.posicao)
            print('Hora: ',dados[0])
            print('Open: ',dados[1])
            print('High: ',dados[2])
            print('Low: ',dados[3])
            print('close: ',dados[4])
            print('acao: ',acao)
            print('stop: ',stop)
            print('gain: ',gain)
            print('Comprado: ',self.comprado)
            print('Vendido: ',self.vendido)
            print('Ficha: ',self.ficha)
            print('Compras: ',self.CC)
            print('Vendas: ',self.VV)
            print('Qtd: ',self.contC,self.contV)
            print('##########################')
        recompensa = 0
        if self.comprado:
            recompensa = dados[4] - self.valor 
        if self.vendido:
            recompensa = self.valor - dados[4]
                
        return self.CC,self.VV, self.saida,self.ficha,self.comprado,self.vendido,recompensa
                
    def verifica(self,dados,gain,stop,verbose):
        if self.comprado:
            stopmax = dados[3]-self.valor
            gainmin = dados[2]-self.valor

            if stopmax < stop:
                pontos = self.valor - (-stop)
                self.ficha = 0
                self.vendido = False
                self.comprado = False
                return self.venda(pontos,dados[0],self.ficha,verbose)
                
            if gainmin > gain: 
                pontos = self.valor + gain
                self.ficha = 0
                self.vendido = False
                self.comprado = False
                return self.venda(pontos,dados[0],self.ficha,verbose)
                
            
        if self.vendido:
            stopmax = self.valor - dados[2]
            gainmin = self.valor - dados[3]
            if stopmax < stop:
                pontos = self.valor + (-stop)
                self.ficha = 0
                self.vendido = False
                self.comprado = False
                return self.compra(pontos,dados[0],self.ficha,verbose)
            if gainmin > gain:
                pontos = self.valor - gain
                self.ficha = 0
                self.vendido = False
                self.comprado = False
                return self.compra(pontos,dados[0],self.ficha,verbose)
        return self.posicao

    def imprimi(self,pmax,pmin):
        print('---------------------------')
        print('valor da compra: ',self.valor)
        print('pontuacao máxima: ',pmax)
        print('pontuacao mínima: ',pmin)
        print('---------------------------')
    def recompensa(self,dados):
        return dados
    def compra(self,entrada,hora,ficha,verbose):
        if ficha == 1:
            self.valor = entrada
            self.contC += 1 
            self.hora = hora
            
            if verbose == 1:
                print('        ')
                print('********************')
                print('COMPRA: ',entrada)
                print('********************')
                print('        ')
            return 1
        else:
            if verbose == 1:
                print('        ')
                print('********************')
                print('VENDIDO: ',self.valor)
                print('COMPRA: ',entrada)
                print('Lucro: ',self.valor - entrada)
                print('inicio: ',self.hora,' termino: ',hora)
                print('Duracao: ',self.Duracao(self.hora,hora))
                print('recompensas: ',self.rewards)
                print('********************')
                print('        ')
            result = self.valor - entrada
            tempo = self.Duracao(self.hora,hora)
            self.VV.append(result)
            self.ficha = 0
            self.saida = self.saida.append({'tipo': 'venda',
                                            'entrada':self.valor,
                                            'saida':entrada,
                                            'ganho': result,
                                            'inicio':self.hora,
                                            'fim':hora,
                                            'duracao': tempo,
                                            'ganhofinal': result - tempo,
                                            'acumulado': sum(self.saida.ganhofinal),
                                            'recompensas': self.rewards}, ignore_index=True)
            self.rewards = []
            return 2
        # return 1
        
    def venda(self,entrada,hora,ficha,verbose):
        if ficha == 1:
            self.valor = entrada
            self.contV += 1
            self.hora = hora
            
            if verbose == 1:
                print('        ')
                print('********************')
                print('VENDA: ',entrada)
                print('********************')
                print('        ')
            return 3
        else:
            if verbose == 1:
                print('        ')
                print('********************')
                print('COMPRADO: ',self.valor)
                print('VENDA: ',entrada)
                print('LUCRO: ',entrada - self.valor)
                print('inicio: ',self.hora,' termino: ',hora)
                print('Duracao: ',self.Duracao(self.hora,hora))
                print('recompensas: ',self.rewards)
                print('********************')
                print('        ')
            result = entrada - self.valor
            tempo = self.Duracao(self.hora,hora)
            self.CC.append(result )
            self.ficha = 0
            self.saida = self.saida.append({'tipo': 'compra',
                                            'entrada':self.valor,
                                            'saida':entrada,
                                            'ganho': result,
                                            'inicio':self.hora,
                                            'fim':hora,
                                            'duracao': tempo,
                                            'ganhofinal': result - tempo,
                                            'acumulado': sum(self.saida.ganhofinal),
                                            'recompensas': self.rewards}, ignore_index=True)
            self.rewards = []
            return 4
    def alvo(self,stop,gain,dados):
        # print('Hora: ',dados[0])
        # print('Open: ',dados[1])
        # print('High: ',dados[2])
        # print('Low: ',dados[3])
        # print('close: ',dados[4])
        x1 = entrada -self.valor
        if self.comprado == true:
            return 0
            
        
    def Duracao(self,base1a,base2a):
        base1 = float(base1a.split(':')[1])
        base2 = float(base2a.split(':')[1])
        base3 = float(base1a.split(':')[0])
        base4 = float(base2a.split(':')[0])
        if base1 > base2:
            tempo = ((60*(base4-base3)) - base1)+ base2
        else:
            tempo = base2 - base1
        if base3 > 17 or base4 > 17 or tempo > 50:
            tempo = 0
        # print((base4-base3),base1,base2,tempo)
        return tempo
    
    def reset(self):
        self.saida = self.saida.drop([i for i in range(len(self.saida))])
        self.ficha = 0
        self.cont = 0
        self.VV = []
        self.CC = []
        self.rewards = []
        self.comprado = False
        self.vendido = False


        
        


