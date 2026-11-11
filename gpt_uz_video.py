import torch

with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) #uzimamo set slova i znakova iz teksta stavljamo ih u listu i sortiramo, chars pokazuje na mjesto u memoriji gdje se nalazi
#vocab_size = len(chars) #printamo velicniu nasih spremljenih slova odnosno znakova i slova
#print(''.join(chars))
#print(vocab_size)

#kreirali smo pretvorbu iz znakova u brojeve odnosno integere
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #uzima string, izlaz je lista integera
decode = lambda l:''.join([itos[i] for i in l]) #decoder uzima listu integera, izlaz je string
print(encode("hi there"))
print(decode(encode("hi there")))

#u ovom dijelu endokiramo cijeli input.txt i spremamo ga u torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000]) #ispisali smo prvih 1000 znakova ali enkodiranih

#razdvojili smo input.txt na podatke za treniranje sto ce biti 90%,a ostatak je za test validacije, odnosno ostatak bi samostalno trebao napisati gptmini
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]