import os

f = open('/home/liuliang/Desktop/alpha_pictures/kisspng/list.txt')
ns = f.readlines()

i = 1
for n in ns:
  n = n.strip()
  n = '/home/liuliang/Desktop/alpha_pictures/kisspng/pictures/' + n
  nn = '/home/liuliang/Desktop/alpha_pictures/kisspng/pictures/{:0>5}.png'.format(i)
  print(n)
  print(nn)
  os.rename(n, nn)
  i = i + 1
