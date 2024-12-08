#  倒過來的3x3乘法表
x=3
while x>=1:
    y = 3
    while y>=1:
        print(x,"乘以",y,"=",x*y)
        y=(y-1)
    x = x-1
    
exit()
    
total =0
x = 1 
while   x<=10:
    if x == 6 or x == 9:
        print(x)
    else:
        total = total + x
    x = x + 1
print(total)    
    