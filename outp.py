
import numpy as np
'''lst=[1,2]
for i in range(0,3):
	#print(lst[i])
	lst[i]=i

print(lst)
#d = [num * 2 for num in lst]
#print(d)
'''

'''from collections import Counter
counter=Counter()
counter[1]=1
counter[2]=2
c=5
counter[c]=
'''
'''import numpy as np
a=np.arange(0,5)
print([-10]+a.tolist()+[-12])

cars = [(3,1), (5,2), (4,3)]
a= [x for x,y in enumerate(cars) if y[1]==2]
print(a[0])

import numpy as np
ar=[]
for i in range(2,302):
    arr=np.arange(0,i)
    ar1=np.arange(0,i)
    np.random.shuffle(arr)
    z=[[[-10]+arr.tolist()+[-12],[-10]+ar1.tolist()+[-12]]]#to convert numpy to a list do this its always more convenient i guess when you're not dealing wih super complex calculations
    ar=ar+z
#print(ar[5])
#print(k for k in range(0,3))
#source,target = zip(*ar) will need to add a couplle of for loops here
#cannot take numpy array anywhere
#source = [[ar[i][0] for i in range[0,50]]] why does this not work
source,target = zip(*[[ar[i][k] for k in range(0,2)] for i in range(0,300)])
a=list(source)
a[0][1]=5
print(a)

a=[1,2,3]
for i in a:
	i=1
print(a)

sourcei=[[1,2,3],[1,5,4]]
y1=[[1,1],[2,2],[3,3],[4,4],[5,5]]
for i in range(len(sourcei)):
    for j in range(len(sourcei[i])):
        sourcei[i][j]=[x for x,y in enumerate(y1) if y[1]==sourcei[i][j]][0]
print(sourcei)        '''
'''
ar=[]
for i in range(2,302):
    arr=np.arange(0,i)
    ar1=np.arange(0,i)
    np.random.shuffle(arr)
    z=[[[-10]+arr.tolist()+[-12],[-10]+ar1.tolist()+[-12]]]#to convert numpy to a list do this its always more convenient i guess when you're not dealing wih super complex calculations
    ar=ar+z
'''
#print(ar[5])
#print(k for k in range(0,3))
#source,target = zip(*ar) will need to add a couplle of for loops here
#cannot take numpy array anywhere
#source = [[ar[i][0] for i in range[0,50]]] why does this not work
'''source,target = zip(*[[ar[i][k] for k in range(0,2)] for i in range(0,300)])#this only works with tuples

from collections import Counter, OrderedDict
counters=Counter()
countert=Counter()
for i in source:
    for j in range(len(i)):
        counters[i[j]]=i[j]
for i in target: 
    for j in range(len(i)):
        countert[i[j]]=i[j] 

y1 = [list(ele) for ele in counters.most_common()]   
#y2 = [list(ele) for ele in OrderedDict(countert.most_common())]
print(y1)
#print(len(countert.keys()))
sourcei=list(source)
targeti=list(target)
for i in range(len(sourcei)):
    for j in range(len(sourcei[i])):
        sourcei[i][j]=[x for x,y in enumerate(y1) if y[1]==sourcei[i][j]][0]
for i in range(len(targeti)):
    for j in range(len(targeti[i])):
        targeti[i][j]=[x for x,y in enumerate(y1) if y[1]==targeti[i][j]][0]

y1=np.array([1,2,3])
#y= x for x,y in enumerate(y1) if y[1]==-10
print(y1.tolist())
'''

import numpy as np
a=1+1j
b=1/a
print(b)