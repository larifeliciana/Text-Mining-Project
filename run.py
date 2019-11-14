import tr
import datetime
import pickle
from itertools import combinations
import sentiment
if __name__ == '__main__':
    domain = ["skoob", "appStore", "steam", "mercadolivre"]

    k = [500]
    dim = [50]

    lista = [[1,0],[0,1]]

    rodadas = len(k)*len(dim)*len(lista)
    conta = 0
    for i in lista:
          for kzinho in k:
                print("faltam "+str(rodadas)+" rodadas")
                rodadas  = rodadas-1

                print("pivots ="+str(kzinho))
                src = i[0]
                dst = i[1]
                time = datetime.datetime.now()
                print("loading....")
                tr.train(domain[src],domain[dst],kzinho,10)
                print("Sent....")
                for d in dim:
                    print(d)
                    sentiment.sent(domain[src],domain[dst],kzinho,10,d,0.1, "logistic","binario", time)
                print(datetime.datetime.now())

    #[(a, b) for a in  for b in lista2]
    """for j in algorithms:
        for n in extraction:
            for i in lista:
                src = i[0]
                dst = i[1]
                print(datetime.datetime.now())
                print("loading....")
                tr.train(domain[src],domain[dst],500,10)
                print("Sent....")
                sentiment.sent(domain[src],domain[dst],500,10,50,0.1, "random")"""
