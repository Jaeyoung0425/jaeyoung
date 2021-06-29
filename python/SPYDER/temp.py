
from torch.utils.data import TensorDataset, DataLoader
from datetime import date, timedelta
import requests
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt  # 맷플롯립사용
import numpy as np  # 넘파이 사용
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import datetime
import random
import re
import pandas as pd
import json
a = 3
b = a+1
print(b)
# %%
marks = [90, 25, 67, 45, 90]
for number in range(len(marks)):
    if marks[number] < 60:
        continue
    print("%d번 학생 축하합니다. 합격입니다." % (number+1))
# %%
add = 0
for i in range(1, 11):
    add = add+i
print(add)
# %%
for i in range(2, 10):
    for j in range(1, 10):
        print(i*j, end " ")
    print('')
# %%
a = [1, 2, 3, 4]
result = []
for num in a:
    result.append(num*3)
print(result)
# %%
a = [1, 2, 3, 4]
result = [num*3 for num in a]
print(result)
# %%
a = [1, 2, 3, 4]
result = [num*3 for num in a if num % 2 == 0]
print(result)
# %%
result = 0
i = 1
while i <= 1000:
    if i % 3 == 0:
        result += i
    i += 1
print(result)
# %%
i = 0
while True:
    i += 1  # while문 수행 시 1씩 증가
    if i > 5:
        break
    print('*'*i)
# %%
for i in range(1, 101):
    print(i)
# %%
marks = [70, 60, 55, 75, 95, 90, 80, 80, 85, 100]
total = 0
for mark in marks:
    total += mark
average = total/len(marks)
print(average)
# %%
numbers = [1, 2, 3, 4, 5]
result = []
for n in numbers:
    if n % 2 == 1:
        result.append(n*2)
# %%


def add(a, b):
    return a+b


a = 3
b = 4
c = add(a, b)
print(c)
# %%


def add(a, b):
    return a+b


print(add(3, 4))
# %%


def say():
    return 'Hi'


print(say())

a = say()
print(a)
# %%
a = add(3, 4)
print(a)
# %%


def add(a, b):
    return a+b


result = add(a=3, b=7)
print(result)
# %%


def add_many(*args):
    result = 0
    for i in args:
        result = result+i
    return result


result = add_many(1, 2, 3)
print(result)
result = add_many(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
print(result)
# %%


def add_mul(choice, *args):
    if choice == "add":
        result = 0
        for i in args:
            result = result+i
    elif choice == "mul":
        result = 1
        for i in args:
            result = result*i
    return result


# %%
number = input("숫자를 입력하세요: ")
# %%
number = input("숫자를 입력하세요: ")
print(number)

# %%
a = 123

print(a)
# %%
f = open("C:/doit/새파일.txt", 'w')
f.close()
# %%
f = open("C:/doit/새파일.txt", 'w')
for i in range(1, 11):
    data = "%d번째 줄입니다.\n" % i
    f.write(data)
f.close()
 # %%
  for i in range(1, 11):
       data = "%d번째 줄입니다.\n" % i
        print(data)
# %%
f = open("C:/doit/새파일.txt", 'r')
while True:
    line = f.readline()
    if not line:
        break
    print(line)
f.close()
# %%
while 1:
    data = input()
    if not data:
        break
    print(data)
# %%
f = open("foo.txt", 'w')
f.write("Life is too short, you need python")
f.close()
with open("foo.txt", "w") as f:
    f.write("Life is too short, you need python")
    # %%


def add(a, b):
    return a+b


a = 3
b = 4
c = add(a, b)
print(c)
 # %%

# %%


def avg_numbers(*args):
    result = 0
    for i in args:
        result += i
    return result/len(args)


print(avg_numbers(1, 2))
# %%


def is_odd(number):
    if number % 2 == 1:
        return True
    else:
        return False


print(is_odd(4))
print(is_odd(3))
# %%


def avg_numbers(*args):
    result = 0
    for i in args:
        result += i
    return result/len(args)


print(avg_numbers(1, 2))
print(avg_numbers(1, 2, 3, 4, 5))
# %%
input1 = input("첫번째 숫자를 입력하세요:")
input2 = input("두번째 숫자를 입력하세요:")

total = input1+input2
print("두 수의 합은 %s 입니다." % total)
# %%
print("you""need""python")
print("you"+"need"+"python")
# %%
f1 = open("test.txt", 'w')
f1.write("Life is too short!")
f1.close()

f2 = open("test.txt", 'r')
print(f2.read())
f2.close()
# %%
print("you""need""python")
print("you"+"need"+"python")
# %%
f1 = open("test.txt", 'w')
f1.write("Life is too short!")

f2 = open("test.txt", 'r')
print(f2.read())
# %%
result = 0


def add(num):
    global result
    result += num
    return result


print(add(3))
print(add(4))
# %%
result1 = 0
result2 = 0


def add1(num):
    global result1
    result1 += num
    return result1


def add2(num):
    global result2
    result2 += num
    return result2


print(add1(3))
print(add1(4))
print(add2(3))
print(add2(7))
# %%


class FourCal:
    pass


a = FourCal()
print(type(a))
# %%


class Fourcal:
    def setdata(self, first, second):
        self.first = first
        self.second = second
# %%


def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n-1)+fib(n-2)


print(fib(5))
# %%


def fib(n):
    result = []

    first = 1
    second = 1
    third = first+second
    for i in range(2, num):

    return result


print(fib(0))
# %% 정답이긴 한데 이해가 안됨


def fib(num):
    result = []
    first = 1
    second = 1
    if(num > 1):
        result.append(first)
        result.append(second)
    for i in range(2, num):
        third = first+second
        result.append(third)
        first = second
        second = third
    return result


print(fib(9))
# %%
for n in range(1, 1000):
    if n % 3 == 0:
        print(n)
        # %%
result = 0
for n in range(1, 1000):
    if n % 3 == 0 or n % 5 == 0:
        result += n
print(result)
# %%
result = 0
for n in range(1, 1000):
    if n % 3 == 0 or n % 5 == 0:
        result += n
print(result)
# %%
sec = 0
for hour in range(24):
    for min in range(60):
        if "3" in str(hour)+str(min):
            sec += 60
print(sec)
# %% 3이 나타나는 시간 전부 합하기
second = 0
for h in range(24):
    for m in range(60):
        if '3' in str(h)+str(m):
            second += 60
print(second)
# %% 정수 배열
list = [-1, 1, 3, -2, 2]
alist = []
blist = []
for i in list:
    if i < 0:
        alist.append(i)
    elif i > 0:
        blist.append(i)
print(alist+blist)
# %% 피보나치 수열


def fib(n):
    if n <= 0:
        return 0
    elif n <= 1:
        return n
    else:
        return fib(n-2)+fib(n-1)


def fib_num(n):
    for n in range(n):
        print(fib(n), end=' ')


fib_num(20)
# %%


def fib(n):
    if n <= 0:
        return 0
    elif n <= 1:
        return n
    else:
        return fib(n-2)+fib(n-1)


def fib_num(n):
    for n in range(n):
        print(fib(n), end=' ')


fib_num(20)
# %% 1~1000에서 각 숫자의 개수 구하기
count = {x: 0 for x in range(0, 10)}

for x in range(1, 1001):
    for i in str(x):
        count[int(i)] += 1

print(count)
# %%
s = ''
for i in range(1000):
    s += str(i+1)

for i in range(10):
    print("{} : ".format(str(i)), s.count("{}".format(i)), "개")
# %%
a = 2
b = 3

s = '구구단 {0}*{1}={2}'.format(a, b, a*b)
print(s)
# %%
s = ''
for x in range(1, 1001):
    s += str(x)
for i in range(10):
    print(str(i)+':%d' % s.count(str(i)), end=' ')
# %%
print("Hello World")
print("Mary's cosmetics")
print('신씨가 소리질렀다. "도둑이야"')
print("안녕하세요.\n만나서\t\t반갑습니다.")  # \n;줄바꿈, \t;탭
print("naver", "kakao", "sk", "samsung", sep=";")
print("naver", "kakao", "sk", "samsung", sep="/")
print("first", end="")
print("second")
print("5/3")
# %%
삼성전자 = 50000
총평가금액 = 삼성전자*10
print(총평가금액)
# %%
s = "hello"
t = "python"
print(s+"!", t)
# %%
a = 2+2*3
print(a)
# %%
a = 125
print(type(a))
# %%
num_str = "720"
num_int = int(num_str)
print(num_int, type(num_int))
# %%
year = "2020"
print(int(year)-3)
print(int(year)-2)
print(int(year)-1)
# %%
second = 0
for h in range(24):
    for m in range(60):
        if '3' in str(h)+str(m):
            second += 60
print(second)
# %%


def fib(n):
    if n <= 0:
        return 0
    elif n <= 1:
        return n
    else:
        return fib(n-2)+fib(n-1)


def fib_num(n):
    for n in range(n):
        print(fib(n), end=" ")


fib_num(20)
# %%


def fib(n):
    if n <= 0:
        return 0
    elif n <= 1:
        return n
    else:
        return fib(n-2)+fib(n-1)


def fib_num(n):
    for n in range(n):
        print(fib(n), end=" ")


print(fib_num(11))
# %%
n = int(input())
list = [0, 1]
for i in range(1, n-1):
    list.append(list[i-1], list[i])

print(list)
# %%


def fib(n):
    if n <= 0:
        return 0
    elif n <= 1:
        return n
    else:
        return fib(n-2)+fib(n-1)


def fib_num(n):
    for n in range(n):
        print(fib(n), end=' ')


fib_num(20)
# %%


def fib(n):
    s = []
    for i in range(n):
        if i > 1:

        else:
            s.append(s[i-1]+s[i-2])

    return s


print(fib(5))
# %% 3이 나타나는 시간 전부 합하기
second = 0
for h in range(24):
    for m in range(60):
        if '3' in str(h)+str(m):
            second += 60
print(second)
# %%


def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-2)+fib(n-1)


def fib_list(max_num):
    s = []
    n = 0
    while fib(n) <= max_num:
        s.append(fib(n))
        n += 1
    return s


print(fib_list(2))
# %% 클래스
result = 0


def add(num):
    global result
    result += num
    return result


print(add(4))
print(add(5))
# %%
num = 100
result = str(num)
print(result, type(result))
# %%
data = "15.79"
data = float(data)
print(data, type(data))
# %%
월 = 48584
총금액 = 월*36
print(총금액)
# %%
lang = 'python'
print(lang[0], lang[2])
license_plate = "24가 2210"
print(license_plate[-4:])
string = "홀짝홀짝홀짝"
print(string[::2])
string = "PYTHON"
print(string[::-1])
phone_number = "010-1111-2222"
phone_number1 = phone_number.replace("-", " ")
print(phone_number1)
phone_number = "010-1111-2222"
phone_number1 = phone_number.replace("-", '')
print(phone_number1)
url = "http://sharebook.kr"
url_split = url.split('.')
print(url_split[-1])
# %%
string = 'abcdfe2a354a32a'
string = string.replace('a', 'A')
print(string)
# %%
a = "3"
b = "4"
print(a+b)
print("Hi"*3)
print("-"*80)
t1 = "python"
t2 = "java"
t3 = t1+' '+t2+' '
print(t3*3)
# %%
name1 = "김민수"
age1 = 10
name2 = "이철희"
age2 = 13
print("이름: %s 나이: %d" % (name1, age1))
print("이름: %s 나이: %d" % (name2, age2))
# %%
name1 = "김민수"
age1 = 10
name2 = "이철희"
age2 = 13
print("이름: {} 나이: {}".format(name1, age1))
print("이름: {} 나이: {}".format(name2, age2))
# %%
name1 = "김민수"
age1 = 10
name2 = "이철희"
age2 = 13
print(f"이름: {name1} 나이: {age1}")
print(f"이름: {name2} 나이: {age2}")
# %%
상장주식수 = "5,969,782,550"
컴마제거 = 상장주식수.replace(",", "")
타입변환 = int(컴마제거)
print(타입변환, type(타입변환))
# %%
분기 = "2020/03(E) (IFRS연결)"
print(분기[:7])
# %%
data = "   삼성전자   "
data1 = data.strip()
print(data1)
# %%
print(str(list(range(1, 10001))).count('8'))
print(str(list(range(1, 10001))).count('8'))
# %%
ticker = "btc_krw"
ticker1 = ticker.upper()
print(ticker1)
ticker = "BTC_KRW"
ticker1 = ticker.lower()
print(ticker1)
a = "hello"
a = a.capitalize()
print(a)
# %%
file_name = "보고서.xlsx"
file_name.endswith("xlsx")
print(file_name)
a = "hello world"
a.split()
print(a)
ticker = "btc_krw"
ticker1 = ticker.split("_")
print(ticker1)
# %% 잠온다....자고싶다...라라랄ㄹㄹㄹ라ㅏㅏ
변수 = 100
print(변수+10)
변수 = 200
print(변수+10)
변수 = 300
print(변수+10)
# %%
list = [100, 200, 300]
for n in list:
    print(n+10)
# %%
list1 = ["김밥", "라면", "튀김"]
for 메뉴 in list1:
    print("오늘의 메뉴:", 메뉴)
# %%
list = ["SK하이닉스", "삼성전자", "LG전자"]
for 종목명 in list:
    print(len(종목명))

# %%
list = ["dog", "cat", "parrot"]
for 동물 in list:
    print(동물, len(동물))
# %%
list = [1, 2, 3]
for n in list:
    # %%
list = ["가", "나", "다", "라"]
for 변수 in list[1:]:
    print(변수)
# %%
list = ["가", "나", "다", "라"]
for 변수 in list[::2]:
    print(변수)
for 변수 in list[::-1]:
    print(변수, end=' ')
# %%
리스트 = [3, -20, -3, 44]
for n in 리스트:
    if n < 0:
        print(n)
# %%
list = [3, 100, 23, 44]
for n in list:
    if n % 3 == 0:
        print(n)
# %%
list = [13, 21, 12, 14, 30, 18]
for n in list:
    if (n < 20) and (n % 3 == 0):
        print(n)
# %%
list = ["I", "study", "python", "language", "!"]
for 변수 in list:
    if len(변수) >= 3:
        print(변수)
# %%
list = ["A", "b", "c", "D"]
for 변수 in list:
    if 변수.isupper():
        print(변수)
# %%
list = ['hello.py', 'ex01.py', 'intro.hwp']
for 변수 in list:
    split = 변수.split(".")
    print(split[0])

변수 = "abcdef"
print(변수.split("c"))
# %%
list = ['intra.h', 'intra.c', 'define.h', 'run.py']
for 변수 in list:
    split = 변수.split(".")
    if split[1] == "h":
        print(변수)
# %%
list = ['intra.h', 'intra.c', 'define.h', 'run.py']
for 변수 in list:
    split = 변수.split(".")
    if (split[1] == "h") or (split[1] == "c"):
        print(변수)
# %%
for n in range(100):
    print(n, end=" ")
# %%
for x in range(2002, 2051, 4):
    print(x)
# %%
for n in range(3, 31, 3):
    print(n)
# %%
for i in range(100):
    print(99-i, end=" ")
# %%
for n in range(10):
    print(n/10)
# %%
for i in range(1, 10):
    print(3, "x", i, "=", 3*i)
# %%
for i in range(1, 10, 2):
    print(3, "x", i, "=", 3*i)
# %%
hab = 0
for i in range(1, 11):
    hab += i
print("합 :", hab)
# %%


def d(n):
    for i in str(n):
        n += int(i)
    return n


a = set((range(1, 5001)))
b = set()
for i in a:
    b.add(d(i))
print(sum(a-b))
# %%


def d(n):
    for i in str(n):
        n += int(i)
    return n


a = set(range(1, 5001))
b = set()
for i in a:
    b.add(d(i))
print(sum(a-b))

# %%

a = 1


def ddd():
    c = 1
    return c


def ccc():
    b = 1
    return b


def main():
    t = ddd() + ccc() + a

    return t


print(ac())

# %%
movie_rank = ["닥터 스트레인지", "스플릿", "럭키"]
movie_rank.append("배트맨")
print(movie_rank)
# %%
movie_rank = ['닥터 스트레인지', '스플릿', '럭키', '배트맨']
movie_rank.insert(1, "슈퍼맨")
print(movie_rank)
# %%
movie_rank = ['닥터 스트레인지', '슈퍼맨', '스플릿', '럭키', '배트맨']
del movie_rank[3]
print(movie_rank)
# %%
movie_rank = ['닥터 스트레인지', '슈퍼맨', '스플릿', '배트맨']
del movie_rank[2]
del movie_rank[2]
print(movie_rank)
# %%
lang1 = ["C", "C++", "JAVA"]
lang2 = ["Python", "GO", "C#"]
langs = lang1+lang2
print(langs)
# %%
nums = [1, 2, 3, 4, 5, 6, 7]
print("max: ", max(nums))
print("min: ", min(nums))
# %%
nums = [1, 2, 3, 4, 5]
print(sum(nums))
print(len(nums))
# %%
nums = [1, 2, 3, 4, 5]
average = sum(nums)/len(nums)
print(average)
# %%
price = ['20180728', '100', '130', '140', '150', '160', '170']
print(price[1:])
# %%
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(nums[::2])
print(nums[::-1])
# %% join; 리스트를 문자열로
interest = ['삼성전자', 'LG전자', 'Naver', 'SK하이닉스', '미래에셋대우']
print("\n".join(interest))
# %%
string = "삼성전자/LG전자/Naver"
interest = string.split("/")
print(interest)
# %%
data = [2, 4, 3, 1, 5, 10, 9]
data.sort()
print(data)
# %%
my_variable = ()
print(type(my_variable))
# %%
interest = ('삼성전자', 'LG전자', 'SK Hynix')
data = list(interest)
print(data)
# %% temp=임시로 저장할 변수
temp = ('apple', 'banana', 'cake')
a, b, c = temp
print(a, b, c)
# %%
data = tuple(range(2, 100, 2))
print(data)
# %%
temp = {}
# %%
ice = {"메로나": 1000, "폴라포": 1200, "빵빠레": 1800}
print(ice)
# %%
ice = {"메로나": 1000, "폴라포": 1200, "빵빠레": 1800}
ice["죠스바"] = 1200
ice["월드콘"] = 1500
print(ice)
# %%
ice = {'메로나': 1000,
       '폴로포': 1200,
       '빵빠레': 1800,
       '죠스바': 1200,
       '월드콘': 1500}
print("메로나 가격: ", ice["메로나"])
# %%
inventory = {"메로나": [300, 20],
             "비비빅": [400, 3],
             "죠스바": [250, 100]}
print(inventory)
# %%
print(inventory["메로나"][0], "원")
print(inventory["메로나"][1], "개")
# %%
icecream = {'탱크보이': 1200, '폴라포': 1200, '빵빠레': 1800, '월드콘': 1500, '메로나': 1000}
ice = list(icecream.keys())
print(ice)
# %%
icecream = {'탱크보이': 1200, '폴라포': 1200, '빵빠레': 1800, '월드콘': 1500, '메로나': 1000}
price = list(icecream.values())
print(price)
# %%


def d(n):
    for i in str(n):
        n += int(i)
    return n
# %%


a = set((range(1, 5001)))
b = set()
for i in a:
    b.add(d(i))
print(sum(a-b))
# %%
print(fib_list(10))
print(d(10))
# %%


def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-2)+fib(n-1)


def fib_list(max_num):
    s = []
    n = 0
    while fib(n) <= max_num:
        s.append(fib(n))
        n += 1
    return s


print(fib_list(10))
# %%


def d(n):
    num_list = []
    for i in range(1, n+1):
        if n % i == 0:
            num_list.append(i)
    return num_list


print(d(10))
# %% 함수안에 함수


def two(n):
    fib_list_first = fib_list(n)
    result_list = []
    for temp_elem in fib_list_first:
        temp_data = d(temp_elem)
        result_list.append(temp_data)
    print(result_list)
    return result_list
# %%


def two(n):
    fib_list_first = fib_list(n)
    result_list = []
    for temp_elem in fib_list_first:
        temp_data = d(temp_elem)  # d(temp_element)=각 항목의 약수 구하기
        result_list.append(temp_data)
    print(result_list)
    return result_list


two(10)
# %%


class Foo:
    pass


obj = Foo()
obj.foo = 2
print(obj.foo)
# %% 8 counting
total = 0
for i in range(1, 10001):
    for n in str(i):
        if n == '8':
            total += 1
print(total)
# %% yayy!!!
total = 0
for i in range(1, 10001):
    for n in str(i):
        if n == '8':
            total += 1
print(total)
# %%

url = 'https://api.nasdaq.com/api/calendar/splits'

headers = {
    'referer': 'https://www.nasdaq.com/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36'
}
params = {
    'date': '2021-01-21'}


# %%

response = requests.get(url, headers=headers, params=params)

# %%
print(type(response.text))
print(type(data_df.executionDate))
# %%
text = json.loads(response.text)

# %%
print(text['data'])

# %%

print(text['data']['rows'])


# %%

print(type(text['data']['rows']))
# %% 표표표표표표표표표표표푶표표표표표표표표표표푶

data_df = pd.DataFrame(text['data']['rows'])
print(data_df)
# %% TQQQ 조건에 맞는 표 출력
print(data_df.loc[1, :])

# %%
print(response.text['data'])


# %%
# 숙제
# symbol의 첫 글자가 A,B,C,D,E,F,G 인것을 가져오세요
data = data_df['symbol'].str.startswith(("A", "B", "C", "D", "E", "F", "G"))
# %%
print(data_df)
# %%
print(data_df['payableDate'])

# %%
print(data_df['payableDate'][:2])
# %%
print(data_df['ratio'].str.startswith('1 : '))

# %%
temp_data = data_df.loc[26, :]
# %%
print(temp_data['ratio'].find(':'))

point = temp_data['ratio'].find(':')
# %% 1:3이상인 것만 가져오기

index_list = list(data_df.index)
result_list = []

for temp_index in index_list:
    print(temp_index)
    temp_data = data_df.loc[temp_index, :]
    print(temp_data['ratio'])
    if temp_data['ratio'].find(':') == -1:
        pass

    else:
        point = temp_data['ratio'].find(':')
        if float(temp_data['ratio'][point+1:])/float(temp_data['ratio'][:point]) > 1:
            print(data_df.loc[temp_index, :])

            result_list.append(temp_index)
            1
        else:
            pass

print(result_list)
# %%
index_list = list(data_df.index)
result_list = []

for temp_index in index_list:
    print(temp_index)
# %%
print(data_df.loc[result_list, :])


# %%
print('333:33'.find(':'))
print('333:33'[:3])
print('333:33'[3+1:])
# %%


def d(n):
    num_list = []
    for i in range(1, n+1):
        if n % i == 0:
            num_list.append(i)
    return num_list


print(d(10))

# %%  index로 column접근하기

temp_data = data_df.loc[temp_index, :]
print(temp_data['ratio'])
if temp_data['ratio'].find(':') == -1:

    # %%

if data_df.loc[temp_index, :]['ratio'].find(':') == -1:

    # %% ratio 데이터로 가져오기

print(data_df.loc[result_list, :])

# %% 날짜차이가 2이상인 것 가져오기

payable_list = list(data_df.payableDate)
exe_list = list(data_df.executionDate)

# %%
print(payable_list)
# %%
for temp_payable in payable_list:
    temp_p = data_df.loc[temp_payable, :]
    print(temp_p)

# %% ratio

index_list = list(data_df.index)
result_list = []

for temp_index in index_list:

    print(temp_index)
    temp_data = data_df.loc[temp_index, :]
    print(temp_data['ratio'])

    if temp_data['ratio'].find(':') == -1:
        pass

    else:
        point = temp_data['ratio'].find(':')
        if float(temp_data['ratio'][point+1:])/float(temp_data['ratio'][:point]) > 1:
            print(data_df.loc[temp_index, :])

            result_list.append(temp_index)
        else:
            pass

print(result_list)
# %%
print(data_df.loc[result_list, :])
# %%

temp_data = data_df.loc[1, :]
if temp_data['payableDate']:

print(data_df.loc[1, :]['payableDate'])
# %%

index_list = list(data_df.index)
result_list = []

for temp_index in index_list:
    print(temp_index)
    temp_data = data_df.loc[temp_index, :]

    if abs(int(temp_data['payableDate'][3:5])-int(temp_data['executionDate'][3:5])) >= 2:
        print(data_df.loc[temp_index, :])

        result_list.append(temp_index)

    else:
        pass

print(result_list)

# %%
print(data_df.loc[result_list, :])
# %%

index_list = list(data_df.index)
result_list = []

for temp_index in index_list:
    print(temp_index)
    temp_data = data_df.loc[temp_index, :]
    print(temp_data['name'])

    c = temp_data['name']
    if c.count(' ') >= 3:
        print(data_df.loc[temp_index, :])
        result_list.append(temp_index)

    else:
        pass

print(result_list)
# %%
print(data_df.loc[result_list, :])
# %%class


class Jalynne:

    def __init__(self, name, sex, dislike):
        self.name = name
        self.sex = sex
        self.dislike = dislike

    def info(self):
        print('제 이름은', self.name, '입니다')
        print('저는', self.sex, '입니다')
        print('저는', self.dislike, '을 싫어합니다')


# %%
jn = Jalynne('Jalynne',
             'female',
             'salmon'
             )

# %%
jn.info()
# %% 진심 울고싶음
숙제

1. Jalynne 클래스를 만든다 .

2. init에서 다음의 변수를 지정한다.
 - 1. Jalynne의 나이, 성별
  - 2. Jalynne이 점심 식사 후에 간식으로 먹고 싶은 것을 List 의 형태로 저장
   - 3. 2월 15일에서 27일까지의 평일을 list로 저장

3. 몇가지 메소드를 만들어 본다(클래스 안에서 정의된 함수를 메소드라고 함)
 - 1. day_pick 이라는 메소드를 만든다.
   - 메소드의 역할: 2월 15일에서 27일 까지의 날짜(init에서 저장한 List) 를 받아서 하루를  랜덤으로 선택해서 return으로 주는 메소드.

    - 2. dessert_pick 이라는 메소드를 만든다.
    - 메소드의 역할: 간식으로 먹고싶은 것을 List의 형태로 저장한 것을 받아서 그중 한가지를  랜덤으로 선택해서 return으로 주는 메소드

    - 3. dessert_result 이라는 메소드를 만든다
    - 메소드의 역할: 메소드 안에서 day_pick, dessert_pick메소드를 사용해서 2월 15일에서  27일중 평일의 날짜 하나를 return하고, 또 간식으로 먹고싶은 것 하나를 return 하는 메소드.

    - 이때 return은 두가지 값을 한번에 return해야 함.
# %%


# %%
print(datetime.datetime.today())
print(datetime.datetime.now())
# %%

t = ['월', '화', '수', '목', '금', '토', '일']
temp_time = datetime.datetime.now() + datetime.timedelta(days=1)
print(temp_time)
print(t[temp_time.weekday()])
# %%
  self.age = age
   self.sex = sex
    age, sex,
# %%


class Jalynne:
    def __init__(self, snack_list, weekday_list):
        self.snack = snack_list
        self.weekday = weekday_list

    t = ['월', '화', '수', '목', '금', '토', '일']
    temp_time = datetime.datetime.now() + datetime.timedelta(days=1)

    def day_pick(self):
        for i in range():
            if t[temp_time.weekday()] = '토' or '일'
             pass
            else:
                print()

    def dessert_pick(self):
        s = self.snack
        print(random.choice(s))

    def dessert_result(self):
        self.dessrt_pick


# %%
d1 = date(2021, 2, 15)
d2 = date(2021, 2, 27)
delta = d2-d1
for i in range(delta.days+1):
    print(d1+timedelta(days=i))
# %%


class Jalynne:
    def __init__(self, snack):
        # self.weekday=weekday
        self.snack = snack

    def day_pick(self):
        t = ['월', '화', '수', '목', '금', '토', '일']
        temp_time = datetime.datetime.now() + datetime.timedelta(days=1)
        d1 = date(2021, 2, 15)
        d2 = date(2021, 2, 27)
        delta = d2-d1

        day_list = []
        for i in range(delta.days+1):
            if t[(d1+timedelta(days=i)).weekday()] == '토' or t[(d1+timedelta(days=i)).weekday()] == '일':
                pass
            else:
                day_list.append(d1+timedelta(days=i))

        target_date = random.sample(day_list, 1)
        return target_date

    def dessert_pick(self):
        s = self.snack
        print(random.sample(s, 1))

    def dessert_result(self):
        self.day_pick
        print(target_date)


# %%

s = Jalynne(['a', 'b', 'c', 'd', 'e'])
s.dessert_pick()

# %%

d = Jalynne()
d.day_pick()

# %%


class Jalynne:
    def __init__(self, snack):
        # self.weekday=weekday
        self.snack = snack

    def day_pick(self):
        t = ['월', '화', '수', '목', '금', '토', '일']
        temp_time = datetime.datetime.now() + datetime.timedelta(days=1)
        d1 = date(2021, 2, 15)
        d2 = date(2021, 2, 27)
        delta = d2-d1
        day_list = []
        for i in range(delta.days+1):
            if t[(d1+timedelta(days=i)).weekday()] == '토' or t[(d1+timedelta(days=i)).weekday()] == '일':
                pass
            else:
                day_list.append(d1+timedelta(days=i))
        return day_list

    def dessert_pick(self):
        s = self.snack
        print(random.sample(s, 1))

    def dessert_result(self):
        day_list = self.day_pick()
        # print(day_list)
        print(random.sample(day_list, 1))


# %%
s = Jalynne(['a', 'b', 'c', 'd', 'e'])
s.dessert_result()

# %% 완성


class Jalynne:
    def __init__(self, sdate, edate, snack):
        self.sdate = sdate
        self.edate = edate
        self.snack = snack

    def day_pick(self):
        t = ['월', '화', '수', '목', '금', '토', '일']
        temp_time = datetime.datetime.now() + datetime.timedelta(days=1)
        delta = self.edate-self.sdate
        day_list = []
        for i in range(delta.days+1):
            if t[(self.sdate+timedelta(days=i)).weekday()] == '토' or t[(self.sdate+timedelta(days=i)).weekday()] == '일':
                pass
            else:
                day_list.append(self.sdate+timedelta(days=i))
        print((random.sample(day_list, 1)))

    def dessert_pick(self):
        s = self.snack
        print((random.sample(s, 1)))

    def dessert_result(self):
        day_list = self.day_pick()
        s = self.dessert_pick()

        return day_list, s


# %%
s = Jalynne(date(2021, 2, 15),
            date(2021, 2, 27),
    ('greentea_frappuccino_with_java_chip_without_whipped_cream', 'Thank_you', 'James')
)

s.dessert_result()
# %% Jalynne_exercise3
df = pd.read_excel(
    r'/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/Jalynne_excercise3.xlsx', engine='openpyxl')
print(df)

# %%
print(df.loc[3, :])

# %% runcell204
corp_df = pd.read_excel(
    'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/corp_info.xlsx', engine='openpyxl')
print(corp_df)
# %%
print(corp_df)

# %%
corp_df['stock_code'].dropna()
# %% 0번째 행의 정보 출력하기
temp_data = df.loc[0, :]
print(temp_data.keys)
# %%
print(temp_data.values[1])
# %%
print(temp_data['총장'])

# %% stock_code중에서 빈 셀 제외한 dataframe 가져오기
i = corp_df.dropna(subset=['stock_code'])
print(i)

# %% stock_name 가져오기 완성
index_list = list(i.index)
result_list = []

for temp_list in index_list:
    temp_data = i.loc[temp_list, :]
    p = temp_data['stock_name']
    result_list.append(p)
print(result_list)
# %% result_list의 형태는 list
print(type(result_list))
# %% 모든 index의 값 (1)
for i in df.index:
    temp_data = df.loc[i, :]
    print(temp_data.keys)

# %%
print(temp_data)

# %% key값으로 value값 출력하기 (2) 데이터 값 하나하나

values = []
for key in temp_data.keys():
    if str(temp_data[key]) != 'nan':
        values.append(temp_data[key])
print(values)

# %%
print(temp_data)
# %%
index_list = list(df.index)  # Jalynne_exercise3 df의 index_list
# %%
print(result_list)
# %%
print(values)
# %%
print(index_list[:10000])

# %%
# %%

for temp_index in tqdm(index_list):
    temp_data = df.loc[temp_index, :]
    values = list(temp_data.values)
    final = []
    flag = 0
    for a in result_list:
        for b in values:
            if a == b:
                final.append(index_list)
                flag = 1
                break
            else:
                pass
        if flag == 1:
            break
        else:
            pass
# %% RE
for temp_index in tqdm(index_list[:10000]):
    temp_data = df.loc[temp_index, :]
    values = list(temp_data.values)
    final = []
    flag = 0
    for a in result_list:
        for b in values:
            b = b.replace(' ', '')
            if a in b:
                final.append(index_list)
                flag = 1
                break
            else:
                pass
        if flag == 1:
            break
        else:
            pass

# %% Rere try except

for temp_index in tqdm(index_list[:50000]):
    temp_data = df.loc[temp_index, :]
    values = list(temp_data.values)
    final = []
    flag = 0
    for a in result_list:
        for b in values:
            try:
                b = b.replace(' ', '')
                if a in b:
                    final.append(index_list)
                    flag = 1
                    break
                else:
                    pass
            except:
                pass
        if flag == 1:
            break
        else:
            pass
# %%
print(len(index_list))
# %%
print(index_list[:10000])epdl
# %%

print(index_list)
# %%
print(final)
# %%
print(result_list.index('LG유플러스'))
print(result_list[1167])

# %% True
print('총장' in temp_data.keys())
# %%
# 1. 국적 찾는 함수


def country(nation, df):

    p = df.dropna(subset=['정당'])
    index_listp = list(p.index)
    p = p.loc[index_listp, :]
    p = p.dropna(axis=1, how='all')
    n = p.dropna(subset=['국적'])

    index_list = list(n.index)
    result_list = []

    for temp_index in index_list:
        temp_data = n.loc[temp_index, :]

        if temp_data['국적'] == nation:
            result_list.append(temp_index)
        else:
            pass

    return n.loc[result_list, :]


print(country('대한민국', df))
# %%
# 2. 출생일 특정년도 이상(?) 찾고+ 생년월일 yyyymmdd로 바꾸는 함수


def birth(year, df):

    #     p=df.dropna(subset=['정당'])
    p = df.dropna(subset=['출생일'])
    index_listp = list(p.index)
    p = p.loc[index_listp, :]
    p = p.dropna(axis=1, how='all')
    p = p.dropna(axis=1, thresh=800)

    birth_year = p['출생일'].astype(str).str[:4]
    birth_year = birth_year[birth_year.str.isdecimal()]
    birth_year = birth_year.astype(int)
    birth_year = birth_year[birth_year >= year]
    # index_list1=특정년도 이상(>=)에 해당하는 인덱스 리스트
    index_list1 = birth_year.index.tolist()

    bb = p.loc[index_list1, :]
    index_list1 = list(bb.index)
    result_list1 = []
    result_list11 = []
    for temp_index1 in index_list1:
        temp_data1 = bb.loc[temp_index1, :]

        point = temp_data1['출생일'].find('(')
        point1 = temp_data1['출생일'].find(')')
        target = temp_data1['출생일'][point+1:point1].strip().replace(' ', '')

        if bool(re.match('.*-.*-.*', temp_data1['출생일'][point+1:point1].strip())) == True:
            z = temp_data1['출생일'][point+1:point1].replace('-', '')
            result_list1.append(temp_index1)
            result_list11.append(z)
        else:
            result_list11.append('no result')

    bb['출생일'] = result_list11
    return bb.loc[result_list1, :]


print(birth(1930, df))
# %%
# 3. 이름 한글만 뽑아내기


def names(name, df):
    p = df.dropna(subset=['출생일'])
    index_listp = list(p.index)
    p = p.loc[index_listp, :]
    p = p.dropna(axis=1, how='all')
    p = p.dropna(axis=1, thresh=800)

    result_list2 = []
    index_list2 = list(p.index)

    for temp_index2 in index_list2:

        if name == '한글':

            temp_data2 = p.loc[temp_index2, :].copy()
            temp_data2['_id'] = str(temp_data2['_id'])
            hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
            h = hangul.sub('', temp_data2['_id'])
            # result_list2에 h를 append하는것=temp_data2의 _id중에 한글문자만 append
            result_list2.append(h)

        elif name == '영어':

            temp_data2 = p.loc[temp_index2, :].copy()
            temp_data2['_id'] = str(temp_data2['_id'])
            eng = re.compile('[^a-zA-Z0-9]')
            e = eng.sub('', temp_data2['_id'])
            result_list2.append(e)

        else:
            pass

    p['_id'] = result_list2  # 그렇게 append해서 만든 list를 p['_id']로 치환하기
    return p


print(names('한글', df))
# %%
# 4.  종교 예쁘게 만들기 완성

hangul = re.compile('[^ ㄱ-ㅣ가-힣  ]+')


def religion(df):
    p = df.dropna(subset=['정당'])
    index_listp = list(p.index)
    p = p.loc[index_listp, :]
    p = p.dropna(axis=1, how='all')
    p = p.dropna(axis=1, thresh=800)

    # for1
    index_list3 = list(p.index)
    result_list = []
    for temp_index3 in index_list3:
        try:
            temp_data3 = p.loc[temp_index3, :]
            r = temp_data3['종교'].replace("[", "(").replace("]", ")").replace(
                "{", "(").replace("}", ")").replace(" ", "")
            r = re.sub(r'\([^)]*\)', '', r).replace(')', '').replace('(', '')
            result_list.append(r)
        except:
            result_list.append('no result')

    p['종교'] = result_list

    # for2
    result_list1 = []
    for temp_result in result_list:
        if bool(re.match('.*→.*', temp_result)) == True:
            point = [i.start() for i in re.finditer('→', temp_result)]
            result_list1.append(temp_result[point[-1]+1:].strip())
        else:
            result_list1.append(temp_result)

    p['종교'] = result_list1

    # for3
    result_list2 = []
    for r in result_list1:
        if "개신교" in r:
            r = r.replace(r, "기독교")
            result_list2.append(r)

        elif "성공회" in r:
            r = r.replace(r, "성공회")
            result_list2.append(r)

        elif "카톨릭" in r:
            r = r.replace(r, "카톨릭")
            result_list2.append(r)

        elif "가톨릭" in r:
            r = r.replace(r, "카톨릭")
            result_list2.append(r)

        elif "무교" in r:
            r = r.replace(r, "무교")
            result_list2.append(r)

        elif "무종교" in r:
            r = r.replace(r, "무교")
            result_list2.append(r)

        elif "무신론" in r:
            r = r.replace(r, "무교")
            result_list2.append(r)

        elif "이슬람" in r:
            r = r.replace(r, "이슬람교")
            result_list2.append(r)

        elif "정교" in r:
            r = r.replace(r, "정교회")
            result_list2.append(r)

        elif "없음" in r:
            r = r.replace(r, "no result")
            result_list2.append(r)

        else:
            result_list2.append(r)

    p['종교'] = result_list2

    return p


print(list(religion(df)['종교']))
# %%
# 5. 출생일이 있는 데이터 중 정당을 예쁘게 찾고 학력 정리하기


def party(df):
    p = df.dropna(subset=['출생일'])
    index_listp = list(p.index)
    p = p.loc[index_listp, :]
    p = p.dropna(axis=1, how='all')
    p = p.dropna(axis=1, thresh=800)

    # for1
    index_list3 = list(p.index)
    result_list = []
    for temp_index3 in index_list3:
        try:
            temp_data3 = p.loc[temp_index3, :]
            r = temp_data3['정당'].replace("[", "(").replace("]", ")").replace(
                "{", "(").replace("}", ")").replace(" ", "")
            r = re.sub(r'\([^)]*\)', '', r).replace(')', '').replace('(', '')
            result_list.append(r)
        except:
            result_list.append(r)

    p['정당'] = result_list

    # for 2
    result_list1 = []
    for temp_result in result_list:
        if bool(re.match('.*→.*', temp_result)) == True:
            point = [i.start() for i in re.finditer('→', temp_result)]
            result_list1.append(temp_result[point[-1]+1:].strip())
        else:
            result_list1.append(temp_result)

    p['정당'] = result_list1

    # for 3 특수문자 없애기
    result_list2 = []
    for s in result_list1:
        try:
            s = re.sub(r'[^ ㄱ-ㅣ가-힣A-Za-z]', '', s)
            result_list2.append(s)
        except:
            result_list2.append(s)
    p['정당'] = result_list2

    # for 4 학력 빈칸 다 없앰
    result_list4 = []
    for temp_index3 in index_list3:
        try:
            temp_data3 = p.loc[temp_index3, :]
            s = temp_data3['학력'].replace(" ", "")
            result_list4.append(s)
        except:
            result_list4.append(temp_data3['학력'])

    p['학력'] = result_list4

   # for 5 univ 이름 append하기
    index_list = list(univ_df.index)
    univ_list = []
    for temp_index in index_list:
        temp_data = univ_df.loc[temp_index, :]
        pp = temp_data['_id']
        univ_list.append(pp)

    final = []
    for b in list(p['학력']):
        flag = 0
        try:
            for a in univ_list:
                if a in b:
                    final.append(a)
                    flag = 1
                    break
                else:
                    pass
        except:
            pass

        if flag == 0:
            final.append('no result')

    # print(len(final))
    # print(len(p['학력']))

    p['학력'] = final
    print(final)

    return p

    print(list(index(p['학력'])))


print(party(df))
# %%
  remove_text = '유교(성리학)'
   print(re.sub(r'\([^)]*\)', '', remove_text))
# %%

    # for 4 학력 빈칸 다 없앰
    result_list4 = []
    for temp_index3 in index_list3:
        try:
            temp_data3 = p.loc[temp_index3, :]
            s = temp_data3['학력'].replace(" ", "")
            result_list4.append(s)
        except:
            result_list4.append(temp_data3['학력'])

    p['학력'] = result_list4

   # for 5 univ 이름 append하기
    index_list = list(univ_df.index)
    univ_list = []
    for temp_index in index_list:
        temp_data = univ_df.loc[temp_index, :]
        pp = temp_data['_id']
        univ_list.append(pp)

    final = []
    for b in list(p['학력']):
        flag = 0
        try:
            for a in univ_list:
                if a in b:
                    final.append(a)
                    flag = 1
                    break
                else:
                    pass
        except:
            pass

        if flag == 0:
            final.append('no result')

    # print(len(final))
    # print(len(p['학력']))

    p['학력'] = final
    print(final)

    return p

    print(list(index(p['학력'])))

print(party(df))
# %%
df = pd.read_excel(
    r'/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/Jalynne_excercise3.xlsx', engine='openpyxl')
# %% social event 함수로 만들기


def social_event(df):
    df = df.dropna(axis=1, how='all')
    df_list = list(df.index)
    events = ['대선', '올림픽', '월드컵', '아시안 게임']

    temp_list = []
    for b in tqdm(df.index[:10000]):
        temp_data = df.loc[b, :]

        values = []
        for key in temp_data.keys():
            if str(temp_data[key]) != 'nan':
                values.append(temp_data[key])
        flag = 0
        for e in events:
            for v in values:
                try:
                    v = v.replace(' ', '')
                    if e in v:
                        temp_list.append(b)
                        flag = 1
                        break
                    else:
                        pass
                except:
                    pass
            if flag == 1:
                break
            else:
                pass
    df = df.loc[temp_list, :]
    print(list(df.index))

    df = df.query(
        '`개최 도시`.notnull() or 참가국.notnull() or 경력.notnull() or `참가 선수`.notnull() or 대회.notnull()')
    return df


print(social_event(df))
# %%
 # 콤마 뒤의 것만 가져온 다음에 이어붙이기!!!!!

  result_list8 = []
   for m in list(p['창당']):

        m = m.replace(' ', '')
        point = m.find('년')
        point1 = m.find('월')
        point2 = m.find('일')

        # mm월
        try:
            if len(m[point+1:point1]) == 1:
                m = m[:point+1]+'0'+m[point1-1:]
                result_list8.append(m)
            else:
                result_list8.append(m)
        except:
            pass
    p['창당'] = result_list8

    result_list9 = []
    for mm in list(p['창당']):

        mm = mm.replace(' ', '')
        point = mm.find('년')
        point1 = mm.find('월')
        point2 = mm.find('일')

        # mm월
        try:
            if len(mm[point1+1:point2]) == 1:
                mm = mm[:point1+1]+'0'+mm[point2-1:]
                result_list9.append(mm)
            else:
                result_list9.append(mm)
        except:
            pass
    p['창당'] = result_list9

    # 년월일 없애기
    result_list10 = []
    for mmm in list(p['창당']):
        try:
            mmm = mmm.replace("년", "").replace("월", "").replace("-","").replace("일","").replace("\n","")
        except:
            result_list10.append(mmm)

    p['창당'] = result_list10

# %%
    result_list8 = []
    for temp_elem in list(p['창당']):
        try:
            if temp_elem.find('년') == -1:  # '년'이 없으면
                result_list8.append('no result')  # no result
                continue
            else:  # '년'이 있으면
                temp_elem = temp_elem.replace(
                    ' ', '').strip()  # '년'이 있으면 공백 없애주기
                count = [i.start() for i in re.finditer(
                    '년', temp_elem)]  # 뒤에서 '년'찾아준걸 count로 지정
                temp_elem = temp_elem[max(
                    0, count[len(count)-1]-5):count[len(count)-1]+8]
                # temp_elem = temp_elem.replace(' ','').strip()
                hangul_numb = re.compile('[^가-힣 0-9]+')  # 정규표현식 한글만
                # temp_elem에서 한글빼고 다 없애주기
                temp_elem = hangul_numb.sub('', temp_elem)

                str1_, str2_, str3_ = '', '', ''  # '' 형태를 str1_ str2_, str3_로 지정해주기
                point1 = temp_elem.find('년')  # temp_elem에서 '년'을 찾은 것을 point1
                # '년' 앞에까지 가져온것=str1 = yyyy가 되겠지 아마도
                str1_ = temp_elem[:point1]
                if temp_elem.find('월') != -1:  # if '월'이 있으면
                    # temp_elem 에서 '월'을 찾은것 = point2
                    point2 = temp_elem.find('월')
                    # '년' 뒤에서부터 월까지=str2 는 mm이 되겠지 아마도
                    str2_ = temp_elem[point1+1:point2]
                    if temp_elem.find('일') != -1:  # '월'이 있는 것 중에 if'일'이 있으면
                        # temp_elem 에서 '월'을 찾은것 = point3
                        point3 = temp_elem.find('일')
                        # '월' 뒤에서부터 '일'까지=str3 는 dd가 되겠지 아마도
                        str3_ = temp_elem[point2+1: point3]
                    else:  # '일'이 없으면 pass
                        pass
                else:  # '월'이 없으면 pass
                    pass

                if len(str2_) == 0:  # if mm이 없으면
                    result_list8.append(str1_)  # yyyy만 append
                    continue
                elif len(str2_) == 1:  # else if mm이 한자리면 0추가해주기
                    str2_ = '0' + str2_
                else:  # mm이 있으면 pass
                    pass

                if len(str3_) == 0:
                    result_list8.append(str1_ + str2_)
                    continue
                elif len(str3_) == 1:
                    str3_ = '0' + str3_
                else:
                    pass

                result_list8.append(str1_ + str2_ + str3_)

        except:
            result_list8.append('no result')

    p['창당'] = result_list8

    return p

print(list(party_info(df)['창당']))
# %%

 # [폐막식] mm월로 만들기
  result_list = []
   for aa in list(df['폐막식']):

        try:
            aa = aa.replace(' ', '')

            point = aa.find('년')
            point1 = aa.find('월')
            point2 = aa.find('일')

            # O월을 OO월로 바꾸기
            try:
                if len(aa[point+1:point1]) == 1:
                    aa = aa[:point+1]+'0'+aa[point1-1]
                    result_list.append(aa)
                else:
                    result_list.append(aa)
            except:
                pass

        except:
            result_list.append('no result')

    df['폐막식'] = result_list

    # [폐막식] dd일로 만들기
    result_list1 = []
    for bb in list(df['폐막식']):

        try:
            bb = bb.replace(' ', '')
            point = bb.find('년')
            point1 = bb.find('월')
            point2 = bb.find('일')

            try:
                if len(bb[point1+1:point2]) == 1:
                    bb = bb[:point1+1]+'0'+bb[point2-1:]
                    result_list1.append(bb)
                else:
                    result_list1.append(bb)
            except:
                pass

        except:
            result_list1.append('no result')

    df['폐막식'] = result_list1

    # [폐막식] 년월일 없애기
    result_list2 = []
    for cc in list(df['폐막식']):
        try:
            cc = cc.replace("년", "").replace("월", "").replace("일","")
            result_list2.append(cc)
        except:
            result_list2.append(cc)

    df['폐막식'] = result_list2

# %%
    # [폐막식] mm월 dd일로 만들기
    result_list = []
    for aa in list(df['폐막식']):

        try:
            aa = aa.replace(' ', '')

            point = aa.find('년')
            point1 = aa.find('월')
            point2 = aa.find('일')

            try:
                if len(aa[point+1:point1]) == 1:
                    aa = aa[:point+1]+'0'+aa[point1-1:]

                    if len(aa[point1+2:point2+1]) == 1:  # 월이 한자리 이면서 일이 한자리이면
                        aa = aa[:point1+2]+'0'+aa[point2:]
                        result_list.append(aa)

                    else:  # 월이 한자리이면서 일이 두자리이면
                        result_list.append(aa)

                else:  # 월이 두자리이면
                    aa = aa[:point1+1]+'0'+aa[point2-1:]
                    result_list.append(aa)
            except:
                result_list.append(aa)
        except:
            result_list.append(aa)

    df['폐막식'] = result_list

    # 괄호 없애고 년월일 없애기
    result_list1 = []
    for aaa in list(df['폐막식']):
        try:
            aaa = re.sub(r'\([^)]*\)', '', aaa)
            aaa = aaa.replace("개최되지않음", "").replace("년", "").replace("월","").replace("일","")
            result_list1.append(aaa)
        except:
            result_list1.append(aaa)
    df['폐막식'] = result_list1
# %%
# %%
t = np.array([0., 1., 2.,3.,4.,5.,6.])
print(t)
print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)
# %%
# %%
t = torch.FloatTensor([1, 2])
print(t.mean())
# %%
t = torch.FloatTensor([[1., 2.], [3.,4.]])
print(t.mean(dim=1))
# %%
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
# %%
print(t.max(dim=0))
# %%
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
 # 해당 tensor는 (3x1)의 크기를 가짐
# %%
print(ft.squeeze())
print(ft.squeeze().shape)
# %%
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
# %%
print(ft.view([-1, 3]))  # ft라는 텐서를 (?, 3)의 크기로 변경
print(ft.view([-1, 3]).shape)
# %%
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
# %%
ft = torch.Tensor([0, 1, 2])
print(ft.shape)
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)
# %%
print(ft.view(1, -1))
print(ft.view(1, -1).shape)
# %%
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())
# %%
bt = torch.ByteTensor([True, False, True, False])
print(bt)
print(bt.long())
print(bt.float())
# %%
x = torch.FloatTensor([[1, 2], [3,4]])
y = torch.FloatTensor([[5, 6], [7,8]])
print(torch.cat([x, y], dim=0))
# %%
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))
# %%
x = torch.FloatTensor([[0, 1, 2],[2,1,0]])
print(x)
print(torch.ones_like(x))
print(torch.zeros_like(x))
# %%
x = torch.FloatTensor([[1, 2], [3,4]])
print(x.mul(2.))
print(x)
print(x.mul_(2.))
print(x)
# %%
# 데이터 이해
# 가설 수립
# 손실 계산
# 경사 하강법

# %%
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# %%
torch.manual_seed(1)
# %% 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
print(x_train)
print(x_train.shape)
print(y_train)
print(y_train.shape)
# %% 모델 초기화
W = torch.zeros(1, requires_grad=True)
print(W)
b = torch.zeros(1, requires_grad=True)
print(b)
# %%
# 직선의 방정식에 해당하는 가설 세우기
hypothesis = x_train * W + b
print(hypothesis)
# %%
# 비용 함수 선언하기
cost = torch.mean((hypothesis-y_train)**2)
print(cost)
# %% optimizer 설정
# SGD= 확률적 경사하강법, 경사 하강법의 일종
lr = learning rate
optimizer = optim.SGD([W, b], lr=0.01)
# %%
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 2000  # 원하는만큼 경사 하강법 선택
for epoch in range(nb_epochs+1):
    hypothesis = x_train*W+b

    cost = torch.mean((hypothesis-y_train)**2)

    # cost로 H(x)개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
# %%
# cost.backward()=가중치 W와 편향 b에 대한 기울기
torch.manual_seed(3)
# torch.manual_seed=난수 발생 순서와 값을 동일하게 보장함
for i in range(1, 3):
    print(torch.rand(1))
# %% autograd=자동 미분
# 값이 2인 임의의 스칼라 텐서 w선언하기
# requires_grad=True=텐서에 대한 기울기 저장
w = torch.tensor(2.0, requires_grad=True)
y = w**2
z = 2*y+5

# .backward=해당 수식의 w에 대한 기울기
z.backward()
# w가 속한 수식을  w로 미분한 값
print(format(w.grad))

# %%
torch.manual_seed(1)

# 훈련 데이터
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# %%
# optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))
# %% Dot Product=벡터의 내적
x_train = torch.FloatTensor([[73,  80,  75],
                              [93,  88,  93],
                              [89,  91,  80],
                              [96,  98,  100],
                              [73,  66,  70]])
y_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

hypothesis = x_train.matmul(W)+b
# %%
x_train = torch.FloatTensor([[73,  80,  75],
                              [93,  88,  93],
                              [89,  91,  80],
                              [96,  98,  100],
                              [73,  66,  70]])
y_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))
# %%
torch.manual_seed(1)

# model=nn.Linear(input_dim,output_dim)
# cost=F.mse_loss(prediction,y_train)
#
# %%
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1, 1)
print(list(model.parameters()))
# 0.5153=W , -0.4414=b

# %%
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
nb_epochs = 2000
for epoch in range(nb_epochs+1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost {:.6f}.'.format(
            epoch, nb_epochs, cost.item()
        ))
# %%
new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)
print(pred_y)
print(list(model.parameters()))
# %%

# %%
torch.manual_seed(1)
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3, 1)
print(list(model.parameters()))
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
# %%
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train)  # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
# %%
torch.manual_seed(1)
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3, 1)
print(list(model.parameters()))
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


# %%
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산 (비용 함수=error function 오차 함수)
    cost = F.mse_loss(prediction, y_train)  # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()

    # 비용 함수를 미분하여 gradient 계산
    cost.backward()

    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# %%
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print(pred_y)

# %% 클래스로 파이토치 모델 구현하기
model = nn.Linear(3, 1)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = MultivariateLinearRegressionModel()
# %%

# %%
x_train = torch.FloatTensor([[73,  80,  75],
                              [93,  88,  93],
                              [89,  91,  90],
                              [96,  98,  100],
                              [73,  66,  70]])
y_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        # print(batch_idx)
        # print(samples)
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader),
            cost.item()
        ))
# %%
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
# %%

# %%


class CustomDataset(Dataset):

     # 데이터셋의 전처리
    def __init__(self):
        self.x_data = [[73, 80,75],
                       [93, 88, 93],
            [89, 91, 90],
            [96, 98, 100],
            [73, 66, 70]]
        self.y_data = [[152], [185], [180],[196],[142]]
        print('hoya')

    # 데이터셋의 길이, 총 샘플의 수
    def __len__(self):
        return len(self.x_data)
    # 데이터셋에서 특정 1개의 샘플을 가져오는 함수

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y


# %%
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# %%
model = torch.nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
# %%
nb_epochs = 20
for epoch in range(nb_epochs + 1):
    print(enumerate(dataloader))
    for i, x in enumerate(dataloader):
        print(i)
        print(x)
    # for batch_idx, samples in enumerate(dataloader):
    #   #print(batch_idx)
    #   # print(samples)
    #   x_train, y_train = samples
    #   # H(x) 계산
    #   prediction = model(x_train)

    #   # cost 계산
    #   cost = F.mse_loss(prediction, y_train)

    #   # cost로 H(x) 계산
    #   optimizer.zero_grad()
    #   cost.backward()
    #   optimizer.step()

    #   print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
    #       epoch, nb_epochs, batch_idx+1, len(dataloader),
    #       cost.item(),r))

# %%
%matplotlib inline
# %% w=1, b=0인 그래프 그리기
# np.arrange(3,7,2)=start at 3, stop at 7(exclude 7), 2간격으로 떨어진 수를 반환= array[3,5]


def sigmoid(x):
    return 1/(1+np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0, 0], [1.0, 0.0],':')
plt.title('Sigmoid Function')
plt.show()

# %%
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--') #w값이 0.5일때
plt.plot(x, y2, 'g',)  # w값이 1일때
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0, 0], [1.0, 0.0],':') #가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
￼
# %%
# H(x)=sigmoid(Wx+b)
torch.manual_seed(1)

x_data = [[1, 2], [2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0], [0], [0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)

# torch.zeros=가중치를 0으로 초기화
W = torch.zeros((2, 1), requires_grad=True)  # 크기=2x1
b = torch.zeros(1, requires_grad=True)

# matmul=행렬의 곱
hypothesis = 1/(1+torch.exp(-(x_train.matmul(W)+b)))
print(hypothesis)
print(y_train)

# %%
e = -(y_train[0] * torch.log(hypothesis[0]) +
      (1 - y_train[0]) * torch.log(1 - hypothesis[0]))

print(e)

# %%
torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid())

print(model(x_train))
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor(
            [0.5])  # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train  # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction)  # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(  # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))

# %%
torch.manual_seed(1)
z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
print(hypothesis.sum())
# %%
torch.manual_seed(1)
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)

y = torch.randint(5, (3,)).long()
print(y)
print(y.shape)
print(y.unsqueeze(1))
print(y.unsqueeze(1).shape)
# %%
y = torch.randint(5, (3,)).long()
print(y)
y.unsqueeze(1)
print(y.unsqueeze(1))
# %%
y_one_hot = torch.zeros_like(hypothesis)  # 모든 원소가 0의 값을 가진 3x5행렬 만들기
# %%
print(y_one_hot)
# %%
print(y.unsqueeze(1))
# %%
print(y_one_hot.scatter_(1, y.unsqueeze(1), 1))
print(y_one_hot.shape)
# %%
y_one_hot = torch.zeros_like(hypothesis)
y = torch.randint(3, (3,)).long()

print(y)
y.unsqueeze(1)
print(y.unsqueeze(1))

print(y_one_hot.scatter_(0, y.unsqueeze(1), 2))
# %%
torch.manual_seed(1)

# (8x4)=8개의 샘플, 4개의 특성
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
# 각 샘플에 대한 레이블, 0/1/2값을 가지므로 총 3개의 클래스가 있다고 추론 가능
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)  # floattensor/longtensor=tensor의 자료형
y_train = torch.LongTensor(y_train)

print(x_train.shape)
print(y_train.shape)
print(y_train.unsqueeze(1))
# %%
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print(y_one_hot.shape)
# %%
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    hypothesis = F.softmax(x_train.matmul(W)+b, dim=1)

    cost = (y_one_hot*-torch.log(hypothesis)).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()))

# %%
model = nn.Linear(4, 3)

# f.cross_entropy()를 사용할 거니까 따로 softmax 함수를 가설에 정의하지 않음
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):

    # h(x)
    prediction = model(x_train)

    # cost
    cost = F.cross_entropy(prediction, y_train)

    # cost로 h(x)개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 20 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch,
              nb_epochs, cost.item()))

# %% 이놈의 클래스


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)  # output=3

    def forward(self, x):
        return self.linear(x)


# %%
model = SoftmaxClassifierModel()
# %%

iris_dataset = load_iris()

data = iris_dataset['data']
target = iris_dataset['target']

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.13, random_state=1)
# %%
print(data)
print(target)

# %%
print(a.shape)
print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])
print(iris_dataset.keys())

# %%

x_train = iris_dataset['data']
y_train = iris_dataset['target']

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

y_one_hot = torch.zeros(150, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
# print(y_one_hot.shape)

W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)
# %%
nb_epochs = 1000000
for epoch in range(nb_epochs + 1):

    # H(x) 가설
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)

    # cost 비용 함수
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# %%
model = nn.Linear(4, 3)

# %%
torch.manual_seed(1)
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.13, random_state=1)
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
# %%
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)
# %%
nb_epochs = 10000
for epoch in range(nb_epochs + 1):
    # H(x) 계산

    prediction = model(x_train)
    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 20번마다 로그 출력
    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# %%
# 정답과 비교하기

# print(data) #150개
# print(target) #각 데이터에 해당하는 꽃 종류 0,1,2로 표시
a = F.softmax(model(x_test), dim=1).argmax(dim= 1)
b = y_test

print(a)
print(b)

c = a == b
print(c)
print(torch.sum(c).item())

# %%
# W,b 저장하기,,
torch.save(model.state_dict(),
           'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/jmodel')

# %%
# %%
iris_dataset = load_iris()

data = iris_dataset['data']
target = iris_dataset['target']

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.13, random_state=1)
# %%
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)
# %%
model2 = nn.Linear(4, 3)
# %%
model2.load_state_dict(torch.load(
    'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/jmodel'))
# %%
# print(F.softmax(model2(x_test), dim=1).argmax(dim = 1))
# %%
print(y_test)
# %%
pred = model2(x_test).argmax(dim=1)
# %%
c = pred == y_test
# %%
print(torch.sum(c))
# %%
print(c)
# %%
# %%
x_train = iris_dataset['data']
y_train = iris_dataset['target']
# print(x_train) #행렬
# print(y_train) #레이블
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
# %%
# Linear 4*32 , relu, Linear 32*16, relu, Linear 16*4 = cross entropy loss 층 쌓기,,,
# sigmoid 대신 ReLU 함수 쓴 것
iris = load_iris
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 4))
print(model)
# %%
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
# %%
losses = []
for epoch in range(1000):  # 괄호 안의 횟수만큼 반복하기
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)  # 손실계산해서 손실 출력하기 (output, target)
    loss.backward()  # 비용함수로부터 기울기 구하기
    optimizer.step()  # 함수 실행시 모델의 parameter들이 update됨

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 1000, loss.item()
        ))

    losses.append(loss.item())
# %%
print(losses)
# %% 손실그래프
plt.plot(losses)
# %%
# %% 붓꽃데이터 분류하기 ,,,
iris_dataset = load_iris()
# 데이터 분리하기
x = iris_dataset['data']
y = iris_dataset['target']

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)

# dataset/dataloader = 방대한 data를 mini batch 단위로 처리할 수 있음
ds_train = TensorDataset(x_train, y_train)
ds_test = TensorDataset(x_test, y_test)

# The training loss will be worse when it shuffles the data bc
# there are more combinations of batches. This is why we shuffle, so
# the model doesn't overfit as much to the training data

loader_train = DataLoader(ds_train, batch_size=10, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=10, shuffle=False)

# %% 다층 퍼셉트론
iris = load_iris
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 4))
# %%
# 오차함수 CrossEntropy
loss_fn = nn.CrossEntropyLoss()

# adam=각 파라미터마다 다른 크기의 업데이트를 적용하는 방법
optimizer = optim.Adam(model.parameters(), lr=0.01)
# %%


def train(epoch):
    model.train()  # 학습모드

    # tensor dataset/dataloader=방대한 data를 mini batch 단위로 처리가능
    for data, target in loader_train:

        optimizer.zero_grad()  # optimizer 초기화
        output = model(data)  # 준비한 데이터를 model에 input으로 넣어서 output을 얻음
        loss = loss_fn(output, target)  # model에서 예측한 결과를 loss_fn에 넣음
        loss.backward()  # calculate gradients using back propagation
        optimizer.step()  # update parameters

    print("epoch{}：완료\n".format(epoch))  # 밑에서 epoch 범위 설정해주면 됨


# %%
for epoch in range(1000):
    train(epoch)
# %% test size = 30, batch size = 10
for data, target in loader_test:
    print('=============predict==========')
    print(model(data).argmax(dim=1))
    print('=============answ=============')
    print(target)
# %%


def test():
    model.eval()  # 추론하기
    correct = 0  # 초기값을 0으로 설정해놓기

    # dataloader에서 mini batch를 하나씩 꺼내 추론 ,,
    with torch.no_grad():
        for data, target in loader_test:

            output = model(data)

            # 확률이 가장 높은 레이블을 predicted라고 선언하기
            predicted = output.data.argmax(dim=1)
            # predicted와  target이 일치한 개수 count
            correct += predicted.eq(target).sum().item()

    data_num = len(loader_test.dataset)
    print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(correct,
                                                         data_num, 100. * correct / data_num))


# %%
test()
# %%
model.eval()  # 신경망을 추론 모드로 전환
data = torch.FloatTensor(x_test[0])
output = model(data)  # 데이터를 입력하고 출력을 계산
predicted = output.data.argmax(dim=1)  # 확률이 가장 높은 레이블이 무엇인지 계산
# %%
print("예측 결과 : {}".format(predicted))
# %%
print(output)
print(x_train[0])
print(data[0])

# %% .eq= 같은 원소의 개수 구하기 (equal)
A = torch.FloatTensor([1, 1, 1])
B = torch.FloatTensor([1, 2, 1])
print(A.eq(B).sum().item())

# %%

# %%

url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/news/v2/list"

querystring = {"region": "US", "snippetCount": "100","s":"TSLA"}

payload = "Pass in the value of uuids field returned right in this endpoint to load the next page, or leave empty to load first page"
headers = {
    'content-type': "text/plain",
    'x-rapidapi-key': "5d145445d0msha8457525bd7c38dp15a6dejsne49ff1d44b49",
    'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
}

response = requests.post(
    url, data=payload, headers=headers, params=querystring)
# %%
print(response.text)
# %%
# %%
data = json.loads(response.text)
# %%
print(data['data']['main']['stream'])
# %%
stream = data['data']['main']['stream']
# %%
print(stream[0]['content']['title'])
# %%
symb = 'TSLA'
# %%
sum = 0
for _ in range(len(stream)):
    title = stream[_]['content']['title']
    if 'Tesla' in title:
        sum += 1
        print(title)
# %%
print(sum)
# %%
symbol_dir = 'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/nasdaq_screener_210406.xlsx'
synonym_dir = 'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/Yahoo_US_Stock_list_synonym_2021_0405_v0.5.csv'
# %%
import pandas as pd
symbol_data = pd.read_excel(symbol_dir, engine = 'openpyxl', dtype = 'object') # 대형주 224개
symbol_list = symbol_data['Symbol'] #대형주 파일에서 'symbol' 칼럼만 가져오기 (종목명)
synonym_data = pd.read_csv(synonym_dir, dtype = 'object', sep = ',', encoding = 'latin') # 동의어사전, encoding 하는 이유 = 한글을 못읽어서 
# %%
print('i AM {} and I am {}'.format( 'Jalynne','james' ))
# print(symbol_data)
a = 'Jalynne'
b = 'James'
print(f'i AM {a} and I am {b}')
# %%
from tqdm import tqdm
# %%
index_list = []              
for i in tqdm(symbol_list): 
    for j in range(len(synonym_data)): # len = 행 전체
        if i == synonym_data.loc[j, :]['symbol_yahoo']: # 행 중에서 'symbol_yahoo'에 해당하는 것
            index_list.append(j)
            break
        else:
            pass
# %%
for _ in symbol_list: 
    if _ not in list(synonym_data['symbol_yahoo']):
        print(_) #223개 밖에 없어서 나머지 하나 찾기
# %%
synonym_data = synonym_data.loc[index_list, :] 
#%%
print(synonym_data) 
# %%
synonym_data.loc[20000,'symbol_yahoo'] = 'INCY' 
# %%
print(len(synonym_data))
# %%
synonym_data['symbol_yahoo'] = 'INCY'
# %%
import time
import datetime
#%%
import requests
import json
# %%
# for _ in range(4):
# try:
data_list = []
data_list2 = []
for i in tqdm(symbol_list):
    time.sleep(1)
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/news/v2/list"
   
    querystring = {"region":"US","snippetCount":"28", 's':i}
   
    payload = "Pass in the value of uuids field returned right in this endpoint to load the next page, or leave empty to load first page"
    headers = {
        'content-type': "text/plain",
        'x-rapidapi-key': "3e9559dd4bmsh053c496e4cf3cbbp1158d9jsn459c6c608bfd",
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
        }
   
    response = requests.post(url, data=payload, headers=headers, params=querystring)
    data = json.loads(response.text)
    stream = data['data']['main']['stream']
    
    
    for j in range(len(stream)):
       
        temp_dict = {}
        temp_dict2 = {}
        title = stream[j]['content']['title']
        title = title
        title_lower = title.lower()
        
        mask = (synonym_data['symbol_yahoo'] == i)
       
        temp_data = synonym_data.loc[mask, :]
       
        for column in temp_data.columns:
            try:
                if temp_data[column].values[0] in title:
                    temp_dict['symbol'] = i
                    temp_dict['title'] = title
                    temp_dict['pubDate'] = datetime.datetime.strptime(stream[j]['content']['pubDate'],'%Y-%m-%dT%H:%M:%SZ').strftime('%Y%m%d')
                    temp_dict['hint'] = temp_data[column].values[0]
                    # print(temp_dict)
                    data_list.append(temp_dict)
                    break
            except:
                pass
            
        for column in temp_data.columns:
            try:
                if temp_data[column].values[0].lower() in title_lower:
                    temp_dict2['symbol'] = i
                    temp_dict2['title'] = title
                    temp_dict2['pubDate'] = datetime.datetime.strptime(stream[j]['content']['pubDate'],'%Y-%m-%dT%H:%M:%SZ').strftime('%Y%m%d')
                    temp_dict2['hint'] = temp_data[column].values[0]
                    # print(temp_dict)
                    data_list2.append(temp_dict2)
                    break
            except:
                pass
# except:
#     pass   
     
crawled_date = datetime.datetime.now().strftime('%Y%m%d')
df = pd.DataFrame(data_list)
df.to_excel('C:/Users/JalynneHEO/re{}.xlsx'.format(crawled_date))
df2 = pd.DataFrame(data_list2)    
df2.to_excel('C:/Users/JalynneHEO/relower{}.xlsx'.format(crawled_date))
# time.sleep(3600*24)
# %%
import torch
import torch.nn as nn

#임의의 텐서 만들기
inputs = torch.Tensor(1,1,28,28)
print(inputs.shape)

#%% 선언만 한 것 
#첫번째 합성곱층과 풀링 선언하기
conv1=nn.Conv2d(1,32,3, padding=1)
print(conv1)

#두번째 합성곱층 구현하기
conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
print(conv2)

#맥스풀링 구현하기/정수 하나를 인자로 넣으면 커널 사이즈와 스트라이드가 둘 다 지정됨 
pool=nn.MaxPool2d(2)
print(pool)
#%% 구현체를 연결하여 모델 만들기

#입력을 첫번째 합성곱층을 통과시킨 후 텐서 크기 확인해보기
out=conv1(inputs)
print(out.shape) #32=conv1의 out_channel이 32로 지정되었음/28=패딩1폭, 3x3커널 사용하면 크기가 보존됨
#%%
#맥스풀링 통과 후 텐서 크기 확인해보기
out=pool(out)
print(out.shape) #14=28/2
#%%두번째 합성곱층에 통과한 후 텐서 크기 확인해보기
out=conv2(out)
print(out.shape) #14=패딩을 1폭으로 하고 3x3 커널을 사용했기 때문에 크기가 보존되었음
#%%
out=pool(out)
print(out.shape) #7=14/2
#%% .view=텐서를 펼침
out=out.view(out.size(0),-1) #첫번째 차원인 배치 차원은 그대로 두고 나머지만 펼치기
print(out.shape) #3136=64*7*7 배치 차원을 제외하고 모두 하나의 차원으로 통합
#%% 전결합층(fully-connected layer)통과시키기
fc=nn.Linear(3136,10) #input_dim=3136, output_dim=10 출력층으로 10개의 뉴런을 배치하여 10개 차원의 텐서로 변환하기
out=fc(out)
print(out.shape)
#%%
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
#%%
device='cuda' if torch.cuda.is_available() else 'cpu'

#랜덤 시드 고정
torch.manual_seed(777)

#GPU 사용 가능일 경우 랜덤 시드 고정
if device=='cuda':
    torch.cuda.manual_seed_all(777)
#%% 파라미터 설정
learning_rate=0.001
training_epochs=15
batch_size=100
#%% 데이터셋 정의하기
mnist_train=dsets.MNIST(root='MNIST_data/', #다운로드 경로 지정
                        train=True, #True=훈련 데이터로 다운로드
                        transform=transforms.ToTensor(),#텐서로 변환
                        download=True)
mnist_test=dsets.MNIST(root='MNIST_data/',
                       train=False, #False=테스트 데이터로 다운로드
                       transform=transforms.ToTensor(), #텐서로 변환
                       download=True)
#%% 데이터로더를 사용하여 배치 크기 설정하기
data_loader=torch.utils.data.DataLoader(dataset=mnist_train,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True)
#%% 클래스로 모델 설계하기
class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN,self).__init__()
        
        


#%%
import spacy
#spacy_en = spacy.blank("en")
spacy_en = spacy.load("en_core_web_sm")
#%%
en_text="A Dog Run back corner near spare bedrooms"
#%%
def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]

print(tokenize(en_text))
#%%
import nltk
nltk.download('punkt')
#%%
from nltk.tokenize import word_tokenize
print(word_tokenize(en_text))
#%%
print(en_text.split())
#%%
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"
print(kor_text.split())

#%%
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
#%%
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh
#%%
from konlpy.tag import Kkma  
kkma=Kkma()  
print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
#%%
print(tokenizer.morphs(kor_text))
#%%
import MeCab
m=MeCab.Tagger()
#%%
print(m.all_morphs)
#%%
out=m.parse("사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어")
#%%
sentence = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"
#%%
out = sentence.all_morphs()
#%%
out = m.all_morphs("사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어")
#%%
print(dir(out))
#%%
print(out.split())
#%%
def jel(sentence):
    m=MeCab.Tagger()
    out=m.parse("사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어")
    return [out.split()[i] for i in range(len(out.split())) if i%2 == 0][:-1]
#%%
print(jel(sentence))
#%%
from konlpy.tag import Okt  
okt=Okt()  
token=okt.morphs("나는 자연어 처리를 배운다") 
#%%
print(list(en_text))
#%%
import urllib.request
import pandas as pd
from konlpy.tag import Mecab
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt
#%%
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data = pd.read_table('ratings.txt') # 데이터프레임에 저장
data[:10]
#%%
print('전체 샘플의 수 : {}'.format(len(data)))
#%%
sample_data = data[:100] # 임의로 100개만 저장
#%%
sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 한글과 공백을 제외하고 모두 제거
sample_data[:10]
#%%
import urllib.request
import pandas as pd
#%%
urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
#%%
df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
df.head()
#%%
print('전체 샘플의 개수 : {}'.format(len(df)))
#%%
train_df = df[:25000]
test_df = df[25000:]
train_df.to_csv("train_data.csv", index=False) #index=False -> index를 저장하지 않음
test_df.to_csv("test_data.csv", index=False)
#%%
from torchtext import data # torchtext.data 임포트
from konlpy.tag import Mecab
#%% 필드 정의하기
from torchtext import data
#%%
from torchtext.legacy import data
text = data.Field(sequential=True, #시퀀스 데이터 여부(true가 기본값)
                  use_vocab=True, #단어 집합을 만들 것인지 여부 (true가 기본값)
                  #tokenizer=str.split, #어떤 토큰화 함수를 사용할 것인지 지정 (string.split이 기본값)
                  lower=True, #영어 데이터를 모두 소문자 (false가 기본값)
                  batch_first=True, #미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지(false가 기본값)
                  fix_length=20) #최대 허용 길이, 이 길에 맞춰서 패딩 작업이 진행됨

label = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

#%%
from torchtext.data import TabularDataset
#%%
from torchtext.data import Iterator
#%%
from konlpy.tag import Okt  

token = Mecab("나는 자연어 처리를 배운다")  
print(token)
#%%
def jel(sentence):
    m=MeCab.Tagger()
    out=m.parse("사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어")
    return [out.split()[i] for i in range(len(out.split())) if i%2 == 0][:-1]
#%%
print(jel(sentence))
#%%
import torch
#%%
dog = torch.FloatTensor([1, 0, 0, 0, 0])
cat = torch.FloatTensor([0, 1, 0, 0, 0])
computer = torch.FloatTensor([0, 0, 1, 0, 0])
netbook = torch.FloatTensor([0, 0, 0, 1, 0])
book = torch.FloatTensor([0, 0, 0, 0, 1])
#%% 원-핫 벡터간 코사인 유사도 구하기
print(torch.cosine_similarity(dog, cat, dim=0))
print(torch.cosine_similarity(cat, computer, dim=0))
print(torch.cosine_similarity(computer, netbook, dim=0))
print(torch.cosine_similarity(netbook, book, dim=0))
#%%
#희소표현(sparse representation)=표현하고자 하는 단어의 인덱스 값만 1, 나머지는 전부 0으로 표현
#밀집표현=희소 표현과 반대되는 표현, 벡터의 차원을 단어 집합의 크기로 상정하지 않음
#
#%%
import nltk
nltk.download('punkt')
#%%
import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize
#%%
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")
#%%
targetXML=open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML) #etree.parse()=매개 변수로 전달된xml파일을 구문 분석하기
parse_text = '\n'.join(target_text.xpath('//content/text()'))



#%%
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk import sent_tokenize
#%%
text_sample = 'The Matrix is everywhere its all around us, here even in this room. You can see it out your window or on your television. You feel it when you go to work, or go to church or pay your taxes.'
tokenized_sentences = sent_tokenize(text_sample)
print(tokenized_sentences)
#%%
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
#print(documents)
print('총 샘플 수 :',len(documents))
#%%
news_df=pd.DataFrame({'document':documents})
#print(news_df['document'])
news_df['clean_doc']=news_df['document'].str.replace("[^a-zA-Z]", " ") #특수문자 제거
news_df['clean_doc']=news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
news_df['clean_doc']=news_df['clean_doc'].apply(lambda x: x.lower())

print(news_df['clean_doc'])
#%% false = null값이 없음
print(news_df.isnull().values.any())
#%%
news_df.replace("",float("Nan"),inplace=True) #모든 빈 값을 Null로 변환하고 다시 Null값이 있는지 확인
print(news_df.isnull().values.any()) #True=Null값이 있음
#%%
news_df.dropna(inplace=True) #inplace=true -> 해당 데이터프레임이 정렬된 결과로 바뀜, 연결된 df자체가 바뀜
print('총 샘플 수 :',len(news_df))

#%%
#nan이 있는지 확인하는 코드 = df.isnull().any()
news_df.replace("",float("Nan"),inplace=True) #inplace=true -> 해당 데이터프레임이 정렬된 결과로 바뀜, 연결된 df자체가 바뀜

#%%
x="나는 Jalynne의 멘토지용"
print(x.split())
print(' '.join([w for w in x.split() if len(w)>3]))
#%%
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = tokenized_doc.to_list()
#%%
import nltk
nltk.download('stopwords')
#%%
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = tokenized_doc.to_list()

#print(tokenized_doc)
#%% ???
drop_train = [index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1]
tokenized_doc = np.delete(tokenized_doc, drop_train, axis=0)
print('총 샘플 수 :',len(tokenized_doc))
#%%
 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_doc) #fit_on_texts = 문자 데이터를 입력받아서 리스트의 형태로 변환

word2idx = tokenizer.word_index #tokenizer의 word_index 속성은 단어와 숫자의 키-값 쌍을 포함하는 딕셔너리를 반환함, 반환 시 자동으로 소문자로 변환되고 느낌표/마침표와 같은 구두점은 자동으로 제거됨
idx2word = {v:k for k, v in word2idx.items()} # k,v가 각각 key, value를 받아와서 {v:k}의 key-value가 뒤집힌 dict형식으로 다시 저장하는 코드
encoded = tokenizer.texts_to_sequences(tokenized_doc) 

print(encoded[:2])
#%%
vocab_size = len(word2idx) + 1 
print('단어 집합의 크기 :', vocab_size)
#%%
from tensorflow.keras.preprocessing.sequence import skipgrams
# 네거티브 샘플링
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:10]]
#%%
# 첫번째 샘플인 skip_grams[0] 내 skipgrams로 형성된 데이터셋 확인
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(5):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
          idx2word[pairs[i][0]], pairs[i][0], 
          idx2word[pairs[i][1]], pairs[i][1], 
          labels[i]))
#%%
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Activation, Input
from tensorflow.keras.layers import Dot
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
#%%
embed_size = 100
#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#%%
# 중심 단어를 위한 임베딩 테이블
w_inputs = Input(shape=(1, ), dtype='int32')
word_embedding = Embedding(vocab_size, embed_size)(w_inputs)

# 주변 단어를 위한 임베딩 테이블
c_inputs = Input(shape=(1, ), dtype='int32')
context_embedding  = Embedding(vocab_size, embed_size)(c_inputs)
#%%
dot_product = Dot(axes=2)([word_embedding, context_embedding])
dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
output = Activation('sigmoid')(dot_product)
#%%
model = Model(inputs=[w_inputs, c_inputs], outputs=output)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')
plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True, rankdir='TB')
#%%
for epoch in range(1, 6):
    loss = 0
    for _, elem in enumerate(skip_grams):
        first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [first_elem, second_elem]
        Y = labels
        loss += model.train_on_batch(X,Y)  
    print('Epoch :',epoch, 'Loss :',loss)
    
#%%
import gensim
#%%
f = open('vectors.txt' ,'w')
f.write('{} {}\n'.format(vocab_size-1, embed_size))
vectors = model.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
f.close()
#%%
w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)
#%%
print(w2v.most_similar(positive=['soldiers']))
#%%
print(w2v.most_similar(positive=['doctor']))
#%%

train_data='you need to know how to code'
word_set=set(train_data.split())
vocab={word: i+2 for i, word in enumerate(word_set)} #단어 집합의 각 단어에 고유한 정수 맵핑
vocab['<unk>']=0
vocab['<pad>']=1
print(vocab)
#%%
#단어 집합의 크기만큼의 행을 가지는 테이블 생성
embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9],
                               [ 0.6,  0.1,  0.1]])

#%%
sample='you need to run'.split()
idxes=[] #이 안에 들어가는 건 숫자 = vocab[word]의 위치
for word in sample:
    try:
        idxes.append(vocab[word])
    except KeyError:
        idxes.append(vocab['<unk>'])
idxes=torch.LongTensor(idxes)

lookup_result=embedding_table[idxes, :]
#%% 임베딩 층 사용하기/nn.Embedding()으로 사용하기

train_data='you need to know how to code'
word_set=set(train_data.split()) #중복 제거한 단어들의 집합
word_set = set(train_data.split()) # 중복을 제거한 단어들의 집합인 단어 집합 생성.
vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}  # 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab['<unk>'] = 0
vocab['<pad>'] = 1
#%%
import torch.nn as nn
embedding_layer = nn.Embedding(num_embeddings = len(vocab), 
                               embedding_dim = 3,
                               padding_idx = 1)

#num_embeddings: 임베딩을 할 단어들의 개수, 단어 집합의 크기
#embedding_dim: 임베딩 할 벡터의 차원, 사용자가 정해주는 하이퍼파라미터
#padding_idx: 선택적으로 사용하는 인자, 패딩을 위한 토큰의 인덱스
#%%
print(embedding_layer.weight)
#%%
import torch
import torchtext
from torchtext import data, datasets
#%% 뭐가문제야 왜 안돼,,,
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)
#%%
from gensim.models import KeyedVectors
#%%
word2vec_model = KeyedVectors.load_word2vec_format('eng_w2v')

#%% be mong sa mong ,, 
import numpy as np

#batch_size, timesteps, input_size의 크기의 3D 텐서를 입력으로 받음

timesteps=10 #시점의 수 = 문장의 길이
input_size=4 #입력의 차원 = 단어 벡터의 차원
hidden_size=8 #은닉 상태의 크기 = 메모리 셀의 용량

inputs=np.random.random((timesteps, input_size)) #입력에 해당하는 2d tensor

hidden_state= np.zeros((hidden_size,)) 
#%%
Wx = np.random.random((hidden_size, input_size))  # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치.
Wh = np.random.random((hidden_size, hidden_size)) # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치.
b = np.random.random((hidden_size,)) # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias).
#%%
Wx = np.random.random((hidden_size, input_size))  # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치.
Wh = np.random.random((hidden_size, hidden_size)) # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치.
b = np.random.random((hidden_size,)) # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias).
#%%
print(np.shape(Wx))
print(np.shape(Wh))
print(np.shape(b)) 
#%%
import tensorflow as tf
#%%
import numpy as np

timesteps = 10 # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_size = 4 # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8 # 은닉 상태의 크기. 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_size)) # 입력에 해당되는 2D 텐서

hidden_state_t = np.zeros((hidden_size,)) # 초기 은닉 상태는 0(벡터)로 초기화
# 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.
#%%

total_hidden_states = []

# 메모리 셀 동작
for input_t in inputs: # 각 시점에 따라서 입력값이 입력됨.
  output_t = np.tanh(np.dot(Wx,input_t) + np.dot(Wh,hidden_state_t) + b) # Wx * Xt + Wh * Ht-1 + b(bias)
  total_hidden_states.append(list(output_t)) # 각 시점의 은닉 상태의 값을 계속해서 축적
  print(np.shape(total_hidden_states)) # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
  hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis = 0) 
# 출력 시 값을 깔끔하게 해준다.

print(total_hidden_states) # (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.
#%%
import torch
import torch.nn as nn
#%%
#은닉 상태의 크기 = 대표적인 RNN의 하이퍼파라미터
#입력의 크기 = 매 시점마다 들어가는 입력의 크기
#입력 텐서 = (배치 크기 x 시점의 수 x 매 시점마다 들어가는 입력) 의 크기를 가짐 ????? 이게 머선말이고

input_size = 5 #입력의 크기
hidden_size = 8 #은닉 상태의 크기
#%%
inputs = torch.Tensor(1,10,5)
cell = 
#%%
import gensim

# 구글의 사전 훈련된 Word2Vec 모델을 로드합니다.
model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/GoogleNews-vectors-negative300.bin', binary=True)  
#%%
print(model['dog'])
#%%
print(model.vectors.shape)
#print(model['book']) 
#%%
#각 stock code에 대해서 title을 word to vector를 사용하여 벡터로 만든뒤 합을 하고, 
#각 title의 vector에 대해서 cosine similarity를 구해서 각 title간의 유사도를 구해 보세요 
#%%
stock_df = pd.read_excel('C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/jalyn_excercise.xlsx', engine='openpyxl')
#%%
sentence = 'Airline Stock Roundup: AAL, LUV & Others Post Loss for Q4, UAL in Focus'
#%%
sentence = sentence.split()
#%%
print(sentence)
#%%

#%%
title_list=list(stock_df['title'])
print(sentences) 
#%% 
symbol_list = stock_df['symbol'].unique().tolist() #'symbol'칼럼을 list타입으로 변환
print(symbol_list)
#%%
mask = (stock_df['symbol'] == symbol)
print(mask)
#%%
for symbol in symbol_list[:1]:
    mask = (stock_df['symbol'] == symbol)
    temp_df = stock_df.loc[mask, :] #mask = index랑 비슷 ,,, ?
    
    title_series = temp_df['title']
    
    index_list = list(title_series.index)
    
    for i in range(len(index_list)): #index_list안에 있는 개수(len())를 범위로 설정
        first_index = index_list[i]
        for j in range(i+1,len(index_list)): #두 문장을 비교하는 거니까 i바로 뒤의 문장 = i+1
            second_index = index_list[j] 
            
            vec1 = 0 
            sentence = title_series.loc[i].split() #title_series = title 칼럼 / split()=나눠서 리스트로 만들기
            for _ in range(len(sentence)):
                try:
                    vec1  = vec1 + model[sentence[_]] 
                except:  
                    pass
                    
            vec2 = 0
            sentence2 = title_series.loc[j].split()
            for _ in range(len(sentence2)):
                try:
                    vec2  = vec2 + model[sentence2[_]] 
                except:
                    pass
            
            similarity = cosine_similarity([vec1, vec2])[0,1] 
            if similarity > 0.8:
                print('{} \nand \n{} \nis similar \nsimilarity = {}'.format(title_series.loc[i], title_series.loc[j], similarity))
                print('===========================================')

#%%

print(title_series.loc[1])
#%%
print(0+model['dog'])
#%%
c = 0 
c = c + [1,2,3]
#%%
print([1:4])
#%%
vec=0
for t in title_list:
    sentence = t.split()
    for _ in range(len(sentence)):
        try:
            vec  = vec + model[sentence[_]] 
        except:
            pass
#%%
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity([vec, vec3])) 
#%%
sentence = [model[i] for i in sentence]

#%%
vec3 = 0
for _ in range(len(sentence)):
    try:
        vec3 = vec3 + model[sentence[_]]
    except:
        pass
#%%

#%%
print(vec2)
#%%
print(vec)
#%% 코사인 유사도 구하기
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity([vec3, vec2])) 
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity([vec3, vec2]))
#%%
import numpy as np
#%% word2vec를 학습하는 신경망 만들기,,, using nn.linear ,,,
# ???
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#%%
numpy : n차원 배열 객체와 함께 이러한 배열들을 조작하기 위해 다양한 함수들을 제공함
      : 과학적 분야의 연산을 위한 포괄적인 프레임워크
      : 딥러닝, 변화도(gradient)에 대해서는 알지 못함, 순전파&역전파 단계를 직접 구현할 수 있음
      
pytorch : 신경망을 구성하고 학습하는 과정에서의 자동 미분
nn 패키지 : 신경망 계층들과 거의 동일한 모듈의 집합을 정의함, 신경망을 학습시킬 때 사용하는 손실 함수도 정의함
nn.Module : 더 복잡한 모델을 구성해야 할 때 사용함, 

#%%

#window_size(o),embedding size, epochs, learning rate 
#학습시킬 데이터 가져오기,, 
#


#%%

#아니 이걸 어떻게 써줘야함 ? 
settings = {
	'window_size': 2	# context window +- center word
	'n': 10,		# dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 50,		# number of training epochs
	'learning_rate': 0.01	# learning rate
}
#%%
Implement Process of CBOW
1.data preparation
2. hyperparameters (learning rate, epochs, window size, embedding size)
3. generate training data (one-hot encoding for words)
4. model training (forward, calculate error rate, adjust weights using backpropagation & compute loss)

#%%

class CBOW(nn.Module):
    def __init__(self, D_len, W_len, #window_size ? ): #d_len=단어 개수(vocab size), w_len=차원(embedding dim)
        super().__init__()
        
        #EX) embedding (7,2,input_length = 5) -> 7=단어의 개수, 2 = 임베딩 한 후의 벡터의 크기, 5 = 입력 시퀀스의 길이.. ? what the hell is sequence .,...
         # ??
        self.linear1=nn.Linear(D_len,W_len) 
        self.linear2=nn.Linear(W_len,D_len)
    
    
    #forward = 모델에서 실행되어야하는 계산을 정의함
    def forward(self, sentence, number): #number = 중심단어의 위치,,, 를 하나씩 넣어야되나,, ?
        if number == 0:
            target_sentece = sentece[1:3]
        
        # elif number / if number == 0 으로 잡으면 elif도 있어야 된다는 말인가 ,,??? ㅠ 
        # 근데 embedding을 쓰면 index로 되어서 이걸 할 필요가 없을것ㄱ 같은데 아닌가
                
                        
        input = torch.nn.tanh(input) #input 은 뭐지 ,, ?
        
        output = self.linear2(input)
        output = F.softmax(out, dim = # ) # dim = ? 
        
        return output
    

#%%
# mini batch 해본다,,,,, 

ted_df = pd.read_csv('C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/ted_eng.csv', engine = 'python-fwf', dtype = object)
#%%
print(ted_df)
#%%

ted_df = pd.read_csv('C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/ted_eng.csv', encoding='latin1')
#%%
print('전체 샘플의 개수 : {}'.format(len(ted_df)))

#%%

train_ted_df = ted_df[:150000]
test_ted_df = ted_df[150000:]
#%%
train_ted_df.to_csv("train_data.csv", index=False)
test_ted_df.to_csv("test_data.csv", index=False)
#%% 필드 정의하기
from torchtext import data 

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False, #false가 기본값
                   is_target=True)

#%% WHAT ?????  넌 또 왜 안되니 
from torchtext.data import TabularDataset
# from torchtext.data import Iterator
#%%
import tensorflow as tf
Iterator = tf.data.Iterator
#%%





