"""
幫我寫出9x9乘法表

"""

for i in range(1, 10):
    for j in range(1, 10):
        print(f"{i} x {j} = {i*j}", end="\t")
    print()

"""
幫我寫出大樂透1-49個數字中選出7個數字包含一個特別號
"""
import random 
lotto_numbers = random.sample(range(1, 50), 7)
print(lotto_numbers)


"""
輸入年份換算印出生肖
"""
year = int(input("請輸入您的出生年份: "))

zodiac_signs = ["鼠", "牛", "虎", "兔", "龍", "蛇", "馬", "羊", "猴", "雞", "狗", "豬"]

birth_year = zodiac_signs[(year - 1912 ) % 12]

print(f"您的出生年份為 {year}，生肖為 {zodiac_signs[birth_year]}")





