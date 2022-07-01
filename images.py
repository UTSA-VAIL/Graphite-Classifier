import os

path = './Data/Images/'

i = 0
for file in os.listdir(path):
    os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.png'))
    i = i+1