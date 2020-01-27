import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

lossfile = './out/loss1.pickle'

with open(lossfile, 'rb') as f:
    loss = pickle.load(f)
print(loss)
x = [i*50 for i in range(2000)]
fig = plt.figure()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x, loss)
plt.legend()
plt.show()
