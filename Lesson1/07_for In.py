#!/usr/bin/env

__author__ = "Powen Ko, www.powenko.com"


for x in range(10):  #0,1,2....,9
    print(x)
print("end")


y=range(10)
y=list(y)
print(y)
for x in y:  #0,1,2....,9
    print(x)
print("end")

y=[20,40,50]
for x in y:  #0,1,2....,9
    print(x)
print("end")

"""
for(i=0,i<10;i++):
    print(i)
"""

""" 筆記
x = 0
while x < 3
    print(x)
    x = x + 1

t1 = [0, 1, 2]
for x in t1:
    print(x)
print("end")
"""

#練習
math=[80,70,90,80]
for x in math:
    print(x)
print("end")


t1=range(10,13)  #10,11,12
for x in t1:
    print(x)
print("end")




for x in range(0,10):
    for y in range(0,10):
        print(x,"x",y,"=",x*y)