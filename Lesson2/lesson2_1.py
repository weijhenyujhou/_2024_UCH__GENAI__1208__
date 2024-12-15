def 列印():
    print("1")
    print("1")
    print("1")
    print("1")
    for x in range(2):
        print(x)
        
def 帶多個參數(str1):
    print(str1) 
    print("end")    


def 帶多個參數(str1,str2,x1):
    print(str1)
    print(str2)
    print(x1)  
    print("end1")

def 帶5個參數(a,b,c,d,e,f):
    sum = a + b + c + d + e + f
    print(sum)

#函數 (半徑)回傳圓周率和面積

#帶多個參數("abc","def",10)
#帶5個參數(1,2,3,4,5,6)
#列印()

#台幣轉美金函數
def currency_converter(TWD_dollar):
    exchange_rate = 0.035
    USdollar = TWD_dollar*exchange_rate
    return USdollar

currency_converter(1000)