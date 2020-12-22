import socket, sys
 

class Comunica():
    def __init__(self,HOST,PORT):
        self.HOST = HOST
        self.PORT = PORT

    def createServer(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print('Socket created')
        except:
            print('Failed to create socket.')
            sys.exit()
        
        try:
            s.bind((self.HOST, self.PORT))
            print('Socket bind complete')
        except:
            print('Bind failed')
            sys.exit()
        
        return s

    def runServer(self,s):
        while True:
            try:
                d = s.recvfrom(2048)    
                data = d[0].decode('utf-8')
                addr = d[1]
            except:
                continue
            return self.recebeDados(data),addr
            
            # try:
            #     # s.sendto(data.encode('utf-8'), addr)
            # except:
            #     continue
        
        s.close()
        
    def recebeDados(self,dados):
        d = []
        dados =dados.split(',',26)
        # print(dados)
        for k in range(len(dados)):
            dados[k] = dados[k].replace('\x00', '')
            d1 = float(dados[k])
            d.append(d1)
        return d
    def enviaDados(self,dados,s,addr):
        s.sendto(dados.encode('utf-8'), addr)
        print('enviado: ',dados)
        return 0


# if __name__ == "__main__":

#     HOST = ''    # Host
#     PORT = 8888  # Porta

#     # s = createServer(HOST, PORT)
#     # runServer(s)
