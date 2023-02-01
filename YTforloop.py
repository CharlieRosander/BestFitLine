from numpy import *
import matplotlib.pyplot as plt

# Skapa x,y för mätpunkter och plota dessa

x1 = [1.0, 2.0, 3.0,]#, dtype = float)
y = [2.0, 4.5, 7.0]#, dtype = float)

# Skapa A och Y matriser från mätpunkter

a = []
for i in range(0, len(x1)):
    a.append(x1[i])
    a.append(1)

A = array(a,dtype=float)
print(A)
A = A.reshape(len(x1), 2)
print(A)
Y = array(y,dtype=float)

# print(A)
# print(Y)
# print(a)
# Skapa normalekvation
At = A.transpose()
A = dot(At, A)
Y = dot(At, Y)
# print(A)
# print(Y)
# print(At)
# Slå ihop x och y till en matris som blir en enda ekvation
a = concatenate((A, Y.reshape(-1, 1)), axis=1)
rows = shape(a)[0]
cols = shape(a)[1]
#solution vector to store solutions
x = zeros(cols-1)
# print(a)


for i in range(cols-1): # spot the first column element making the factor
    # and use it to eliminate the index below that index where i made
    # my factor
    for j in range(i + 1, rows): # Dont want to inlude the pivot element
        # a[j,:] row j and all columns
        # a[j,i] = first elemnt in row j
        # a[i, i] = pivot element
        a[j, :] = -(a[j, i]/a[i, i])*a[i, :]+a[j, :]
# print(a)
    # back substitution algorithm
    # bottom right to the top left

for i in arange(rows-1, -1, -1): # iterating from row -1 to row - 1
    x[i] = (a[i, -1] - a[i, 0:cols-1]@x)/a[i, i]  # storing the last element of a in x-vector
#     # @ = matrix multiplication
#     # start at row i and go from 0 to columns - 1 and mulitply everything
#     # with the x-vector

# print(a)
# print(x1)

# Skriv ut "bästa linjen" och plota linjen

k = round(x[0], 2)
m = round(x[1], 2)

plt.plot(x, k * x + m)
plt.scatter(x1, y)
plt.show()