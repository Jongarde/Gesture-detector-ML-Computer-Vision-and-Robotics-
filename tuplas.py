lista = [[(1,2), (3,4)], [(1,2), (5,6)], [(4,8), (5,9)]]

def tienen_elemento_comun(lista):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)-1):
            if set(lista[i]).intersection(set(lista[j])):
                lista.append(set(lista[i]+lista[j]))
                lista.remove(lista[i])
                lista.remove(lista[j])
                
tienen_elemento_comun(lista)
print(lista)


		
