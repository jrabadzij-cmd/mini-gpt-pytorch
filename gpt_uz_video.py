import torch
import torch.nn as nn
from torch.nn import functional as F

with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) #uzimamo set slova i znakova iz teksta stavljamo ih u listu i sortiramo, chars pokazuje na mjesto u memoriji gdje se nalazi
vocab_size = len(chars) #printamo velicniu nasih spremljenih slova odnosno znakova i slova
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

#data loader: batches of chunks of data
block_size = 8
train_data[:block_size+1]
x = train_data[:block_size]
y = train_data[1:block_size+1]
#for t in range(block_size):
#    context = x[:t+1]
#   target = y[t]
#    print(f"when input is {context} the target: {target}")


torch.manual_seed(1337) #nasumicni brojevi postaju ponovljivi, skavi put pri pokretanju koda dobimo iste nasumične pozicije
batch_size = 4 #koliko chunkova obrađujemo paralelno
block_size = 8 #koliko je dugačak svaki chunk

def get_batch(split):
    data = train_data if split == 'train' else val_data #uzima se train ili val dio
    ix = torch.randint(len(data) - block_size, (batch_size,)) #gemeriranje 4 nasumična broja između nula i duljine podataka - veličine chunka
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #za svaku početnu poziciju i izreže komad od 8 znakova(x) i taj isti komad pomaknut za jedno mjesto(y) torch stack slozi cetiri komada jedan ispod drugog u tenzor oblika (4,8)
    return x, y

xb, yb = get_batch('train')
print('intputs: ')
print(xb.shape)
print(xb)
print('targets: ')
print(yb.shape)
print(yb)

print('------')

for b in range(batch_size): #prolazi kroz 4 chunka
    for t in range(block_size): #prolazi kroz 8 pozicija u chunku
        context = xb[b, :t+1] #prvih t+1 znakova chunka
        target = yb[b, t] #sljedeci znak
        print(f"when input is {context.tolist()} the target: {target}")

torch.manual_seed(1337)

class BiagramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) #svaki token direktno cita logits za sljedeci token iz lookup table
    def forward(self, idx, targets=None): #idx i targets su oboje B i T tensor integera

        logits = self.token_embedding_table(idx) #(B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens): #idx je polje (B, T) u trenutnom kontekstu
            logits, loss = self(idx) #prihvaca predikcije
            logits = logits[:, -1, :] #fokusira se samo na zadnji vremenski korak, postaje (B, C)
            probs = F.softmax(logits, dim=-1) #koristimo softmax da dobimo šanse, (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) #uzorak iz distribucije, (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx


m = BiagramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


    
