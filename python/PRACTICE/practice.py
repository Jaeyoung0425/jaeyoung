
a=3
b = a+1
print(b)
#%%
marks=[90,25,67,45,90]
for number in range(len(marks)):
    if marks[number]<60:
        continue
    print("%d번 학생 축하합니다. 합격입니다."%(number+1))
#%%
add=0
for i in range(1,11):
    add=add+i
print(add)
#%%
for i in range(2,10):
    for j in range(1, 10):
        print(i*j, end " ")
    print('')
#%%
a=[1,2,3,4]
result=[]
for num in a:
    result.append(num*3)
print(result)
#%%
a=[1,2,3,4]
result=[num*3 for num in a]
print(result)
#%%
a=[1,2,3,4]
result=[num*3 for num in a if num %2==0]
print(result)
#%%
result=0
i=1
while i<=1000:
    if i%3==0:
        result+=i
    i+=1
print(result)
#%%
i=0
while True:
    i+=1 #while문 수행 시 1씩 증가
    if i>5 : break
    print('*'*i)
#%%
for i in range(1,101):
    print(i)
#%%
marks=[70,60,55,75,95,90,80,80,85,100]
total=0
for mark in marks:
    total+=mark
average=total/len(marks)
print(average)
#%%
numbers=[1,2,3,4,5]
result=[]
for n in numbers:
    if n%2==1:
        result.append(n*2)
#%%
def add(a,b):
    return a+b
a=3
b=4
c=add(a,b)
print(c)
#%%
def add(a,b):
    return a+b
print(add(3,4))
#%%
def say():
    return 'Hi'
print(say())

a=say()
print(a)
#%%
a=add(3,4)
print(a)
#%%
def add(a,b):
    return a+b
result=add(a=3,b=7)
print(result)
#%%
def add_many(*args):
    result=0
    for i in args:
        result=result+i
    return result

result=add_many(1,2,3)
print(result)
result=add_many(1,2,3,4,5,6,7,8,9,10)
print(result)
#%%
def add_mul(choice,*args):
    if choice=="add":
        result=0
        for i in args:
            result=result+i
    elif choice=="mul":
        result=1
        for i in args:
            result=result*i
    return result
#%%
number=input("숫자를 입력하세요: ")
#%%
number=input("숫자를 입력하세요: ")
print(number)

#%%
a=123

print(a)
#%%
f=open("C:/doit/새파일.txt",'w')
f.close()
#%%
f=open("C:/doit/새파일.txt",'w')
for i in range(1,11):
    data="%d번째 줄입니다.\n" %i
    f.write(data)
f.close()    
    #%%
    for i in range(1,11):
        data="%d번째 줄입니다.\n" %i
        print(data)
#%%
f=open("C:/doit/새파일.txt",'r')
while True:
    line=f.readline()
    if not line:break
    print(line)
f.close()
#%%
while 1:
    data=input()
    if not data: break
    print(data)
#%%
f=open("foo.txt",'w')
f.write("Life is too short, you need python")
f.close()
with open("foo.txt","w") as f:
    f.write("Life is too short, you need python")
    #%%
def add(a,b):
    return a+b
a=3
b=4
c=add(a,b)
print(c)    
    #%%

#%%
def avg_numbers(*args):
    result=0
    for i in args:
        result+=i
    return result/len(args)

print(avg_numbers(1,2))
#%%
def is_odd(number):
    if number %2 ==1:
        return True
    else:
        return False
    
print(is_odd(4))
print(is_odd(3))
#%%
def avg_numbers(*args):
    result=0
    for i in args:
        result+=i
    return result/len(args)
print(avg_numbers(1,2))
print(avg_numbers(1,2,3,4,5))
#%%
input1=input("첫번째 숫자를 입력하세요:")
input2=input("두번째 숫자를 입력하세요:")

total=input1+input2
print("두 수의 합은 %s 입니다." %total)
#%%
print("you""need""python")
print("you"+"need"+"python")
#%%
f1=open("test.txt",'w')
f1.write("Life is too short!")
f1.close()

f2=open("test.txt",'r')
print(f2.read())
f2.close()
#%%
print("you""need""python")
print("you"+"need"+"python")
#%%
f1=open("test.txt",'w')
f1.write("Life is too short!")

f2=open("test.txt",'r')
print(f2.read())
#%%
result=0
def add(num):
    global result
    result+=num
    return result

print(add(3))
print(add(4))
#%%
result1=0
result2=0

def add1(num):
    global result1
    result1+=num
    return result1

def add2(num):
    global result2
    result2+=num
    return result2

print(add1(3))
print(add1(4))
print(add2(3))
print(add2(7))
#%%
class FourCal:
    pass
a=FourCal()
print(type(a))
#%%
class Fourcal:
    def setdata(self, first, second):
        self.first=first
        self.second=second
#%%
def fib(n):
    if n<=1:
        return n
    else:
        return fib(n-1)+fib(n-2)
print(fib(5))
#%%
def fib(n):
    result=[]

    first=1
    second=1
    third=first+second
    for i in range(2,num):

    return result
print(fib(0))
#%% 정답이긴 한데 이해가 안됨
def fib(num):
    result=[]
    first=1
    second=1
    if(num>1):
        result.append(first)
        result.append(second)
    for i in range(2,num):
        third=first+second
        result.append(third)
        first=second
        second=third
    return result
print(fib(9))
#%%
for n in range(1,1000):
    if n%3==0:
        print(n)
        #%%
result=0
for n in range(1,1000):
    if n %3==0 or n%5==0:
        result+=n
print(result)
#%%
result=0
for n in range(1,1000):
    if n%3==0 or n%5==0:
        result+=n
print(result)
#%%
sec=0
for hour in range(24):
    for min in range(60):
        if "3" in str(hour)+str(min):sec+=60
print(sec)
#%% 3이 나타나는 시간 전부 합하기
second=0
for h in range(24):
    for m in range(60):
        if '3' in str(h)+str(m):
            second+=60
print(second)
#%% 정수 배열
list=[-1,1,3,-2,2]
alist=[]
blist=[]
for i in list:
    if i<0:
        alist.append(i)
    elif i>0:
        blist.append(i)
print(alist+blist)
#%% 피보나치 수열
def fib(n):
    if n<=0:
        return 0
    elif n<=1:
        return n
    else:
        return fib(n-2)+fib(n-1)
    
def fib_num(n):
    for n in range(n):
        print(fib(n), end=' ')

fib_num(20)
#%%
def fib(n):
    if n<=0:
        return 0
    elif n<=1:
        return n
    else:
        return fib(n-2)+fib(n-1)
    
def fib_num(n):
    for n in range(n):
        print(fib(n), end=' ')

fib_num(20)
#%% 1~1000에서 각 숫자의 개수 구하기
count={ x:0 for x in range(0,10) }

for x in range(1,1001):
    for i in str(x):
        count[int(i)]+=1

print(count)
#%% 
s = ''
for i in range(1000):
    s += str(i+1)

for i in range(10):
    print("{} : ".format(str(i)),s.count("{}".format(i)),"개")
#%%
a=2
b=3

s='구구단 {0}*{1}={2}'.format(a,b,a*b)
print(s)
#%%
s=''
for x in range(1,1001):s+=str(x)
for i in range(10):print(str(i)+':%d'%s.count(str(i)), end=' ')
#%%
print("Hello World")
print("Mary's cosmetics")
print('신씨가 소리질렀다. "도둑이야"')
print("안녕하세요.\n만나서\t\t반갑습니다.") #\n;줄바꿈, \t;탭
print("naver","kakao","sk","samsung",sep=";")
print("naver","kakao","sk","samsung",sep="/")
print("first",end="");print("second")
print("5/3")
#%%
삼성전자=50000
총평가금액=삼성전자*10
print(총평가금액)
#%%
s="hello"
t="python"
print(s+"!",t)
#%%
a=2+2*3
print(a)
#%%
a=125
print(type(a))
#%%
num_str="720"
num_int=int(num_str)
print(num_int,type(num_int))
#%%
year="2020"
print(int(year)-3)
print(int(year)-2)
print(int(year)-1)
#%%
second=0
for h in range(24):
    for m in range(60):
        if '3' in str(h)+str(m):
            second+=60
print(second)
#%%
def fib(n):
    if n<=0:
        return 0
    elif n<=1:
        return n
    else:
        return fib(n-2)+fib(n-1)
    
def fib_num(n):
    for n in range(n):
        print(fib(n),end=" ")
        
fib_num(20)
#%%
def fib(n):
    if n<=0:
        return 0
    elif n<=1:
        return n
    else:
        return fib(n-2)+fib(n-1)
    
def fib_num(n):
    for n in range(n):
        print(fib(n),end=" ")
        
print(fib_num(11))
#%%
n=int(input())
list=[0,1]
for i in range(1,n-1):
    list.append(list[i-1],list[i])

print(list)
#%%
def fib(n):
    if n<=0:
        return 0
    elif n<=1:
        return n
    else:
        return fib(n-2)+fib(n-1)
    
def fib_num(n):
    for n in range(n):
        print(fib(n), end=' ')

fib_num(20)
#%%
def fib(n):
    s=[]
    for i in range(n):
        if i>1:
            
        else:
            s.append(s[i-1]+s[i-2])
            
    return s

print(fib(5))
#%% 3이 나타나는 시간 전부 합하기
second=0
for h in range(24):
    for m in range(60):
        if '3' in str(h)+str(m):
            second+=60
print(second)
#%%
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
#%% 클래스
result=0
def add(num):
    global result
    result+=num
    return result
print(add(4))
print(add(5))
#%%
num=100
result=str(num)
print(result,type(result)) 
#%%
data="15.79"
data=float(data)
print(data,type(data))
#%%
월=48584
총금액=월*36
print(총금액)
#%%
lang='python'
print(lang[0],lang[2])
license_plate="24가 2210"
print(license_plate[-4:])
string="홀짝홀짝홀짝"
print(string[::2])
string="PYTHON"
print(string[::-1])
phone_number="010-1111-2222"
phone_number1=phone_number.replace("-"," ")
print(phone_number1)
phone_number="010-1111-2222"
phone_number1=phone_number.replace("-",'')
print(phone_number1)
url="http://sharebook.kr"
url_split=url.split('.')
print(url_split[-1])
#%%
string='abcdfe2a354a32a'
string=string.replace('a','A')
print(string)
#%%
a="3"
b="4"
print(a+b)
print("Hi"*3)
print("-"*80)
t1="python"
t2="java"
t3=t1+' '+t2+' '
print(t3*3)
#%%
name1="김민수"
age1=10
name2="이철희"
age2=13
print("이름: %s 나이: %d" % (name1,age1))
print("이름: %s 나이: %d" %(name2, age2))
#%%
name1="김민수"
age1=10
name2="이철희"
age2=13
print("이름: {} 나이: {}".format(name1,age1))
print("이름: {} 나이: {}".format(name2,age2))
#%%
name1="김민수"
age1=10
name2="이철희"
age2=13
print(f"이름: {name1} 나이: {age1}")
print(f"이름: {name2} 나이: {age2}")
#%%
상장주식수="5,969,782,550"
컴마제거=상장주식수.replace(",","")
타입변환=int(컴마제거)
print(타입변환,type(타입변환))
#%%
분기="2020/03(E) (IFRS연결)"
print(분기[:7])
#%%
data="   삼성전자   "
data1=data.strip()
print(data1)
#%%
print(str(list(range(1, 10001))).count('8'))
print(str(list(range(1,10001))).count('8'))
#%%
ticker="btc_krw"
ticker1=ticker.upper()
print(ticker1)
ticker="BTC_KRW"
ticker1=ticker.lower()
print(ticker1)
a="hello"
a=a.capitalize()
print(a)
#%%
file_name="보고서.xlsx"
file_name.endswith("xlsx")
print(file_name)
a="hello world"
a.split()
print(a)
ticker="btc_krw"
ticker1=ticker.split("_")
print(ticker1)
#%% 잠온다....자고싶다...라라랄ㄹㄹㄹ라ㅏㅏ
변수=100
print(변수+10)
변수=200
print(변수+10)
변수=300
print(변수+10)
#%%
list=[100,200,300]
for n in list:
    print(n+10)
#%%
list1=["김밥","라면","튀김"]
for 메뉴 in list1:
    print("오늘의 메뉴:",메뉴)
#%%
list=["SK하이닉스","삼성전자","LG전자"]
for 종목명 in list:
    print(len(종목명)) 
    
#%%
list=["dog","cat","parrot"]
for 동물 in list:
    print(동물,len(동물))
#%%
list=[1,2,3]
for n in list:
#%%
list=["가","나","다","라"]
for 변수 in list[1:]:
    print(변수)
#%%
list=["가","나","다","라"]
for 변수 in list[::2]:
    print(변수)
for 변수 in list[::-1]:
    print(변수,end=' ')
#%% 
리스트=[3,-20,-3,44]
for n in 리스트:
    if n<0:
        print(n)
#%%
list=[3,100,23,44]
for n in list:
    if n%3==0:
        print(n)
#%%
list=[13,21,12,14,30,18]
for n in list:
    if (n<20) and (n%3==0):
        print(n)
#%%
list=["I","study","python","language","!"]
for 변수 in list:
    if len(변수)>=3:
        print(변수)
#%%
list=["A","b","c","D"]
for 변수 in list:
    if 변수.isupper():
        print(변수)
#%%
list=['hello.py','ex01.py','intro.hwp']
for 변수 in list:
    split=변수.split(".")
    print(split[0])
    
변수 = "abcdef"
print(변수.split("c"))
#%%
list=['intra.h','intra.c','define.h','run.py']
for 변수 in list:
    split=변수.split(".")
    if split[1]=="h":
        print(변수)
#%% 
list=['intra.h','intra.c','define.h','run.py']
for 변수 in list:
    split=변수.split(".")
    if (split[1]=="h") or (split[1]=="c"):
        print(변수)
#%%
for n in range(100):
    print(n,end=" ")
#%%
for x in range(2002,2051,4):
    print(x)
#%%
for n in range(3,31,3):
    print(n)
#%%
for i in range(100):
    print(99-i,end=" ")
#%%
for n in range(10):
    print(n/10)
#%%
for i in range(1,10):
    print(3,"x",i,"=",3*i)
#%%
for i in range(1,10,2):
    print(3,"x",i,"=",3*i)
#%% 
hab=0
for i in range(1,11):
    hab+=i
print("합 :",hab)
#%%
def d(n):
    for i in str(n):
        n+=int(i)
    return n

a=set((range(1,5001)))
b=set()
for i in a:
    b.add(d(i))
print(sum(a-b))
#%%
def d(n):
    for i in str(n):
        n+=int(i)
    return n

a=set(range(1,5001))
b=set()
for i in a:
    b.add(d(i))
print(sum(a-b))

#%%

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
    
#%%
movie_rank=["닥터 스트레인지","스플릿","럭키"]
movie_rank.append("배트맨")
print(movie_rank)
#%%
movie_rank=['닥터 스트레인지','스플릿','럭키','배트맨']
movie_rank.insert(1,"슈퍼맨")
print(movie_rank)
#%%
movie_rank=['닥터 스트레인지','슈퍼맨','스플릿','럭키','배트맨']
del movie_rank[3]
print(movie_rank)
#%%
movie_rank=['닥터 스트레인지','슈퍼맨','스플릿','배트맨']
del movie_rank[2]
del movie_rank[2]
print(movie_rank)
#%%
lang1=["C","C++","JAVA"]
lang2=["Python","GO","C#"]
langs=lang1+lang2
print(langs)
#%%
nums=[1,2,3,4,5,6,7]
print("max: ",max(nums))
print("min: ",min(nums))
#%%
nums=[1,2,3,4,5]
print(sum(nums))
print(len(nums))
#%%
nums=[1,2,3,4,5]
average=sum(nums)/len(nums)
print(average)
#%%
price=['20180728','100','130','140','150','160','170']
print(price[1:])
#%%
nums=[1,2,3,4,5,6,7,8,9,10]
print(nums[::2])
print(nums[::-1])
#%% join; 리스트를 문자열로
interest=['삼성전자','LG전자','Naver','SK하이닉스','미래에셋대우']
print("\n".join(interest))
#%%
string="삼성전자/LG전자/Naver"
interest=string.split("/")
print(interest)
#%%
data=[2,4,3,1,5,10,9]
data.sort()
print(data)
#%%
my_variable=()
print(type(my_variable))
#%%
interest=('삼성전자','LG전자','SK Hynix')
data=list(interest)
print(data)
#%% temp=임시로 저장할 변수
temp=('apple','banana','cake')
a,b,c=temp
print(a,b,c)
#%%
data=tuple(range(2,100,2))
print(data)
#%%
temp={}
#%%
ice={"메로나":1000,"폴라포":1200,"빵빠레":1800}
print(ice)
#%%
ice={"메로나":1000,"폴라포":1200,"빵빠레":1800}
ice["죠스바"]=1200
ice["월드콘"]=1500
print(ice)
#%%
ice = {'메로나': 1000,
       '폴로포': 1200,
       '빵빠레': 1800,
       '죠스바': 1200,
       '월드콘': 1500}
print("메로나 가격: ", ice["메로나"])
#%%
inventory={"메로나": [300,20],
           "비비빅": [400,3],
           "죠스바": [250,100]}
print(inventory)
#%%
print(inventory["메로나"][0],"원")
print(inventory["메로나"][1],"개")
#%% 
icecream={'탱크보이':1200,'폴라포':1200,'빵빠레':1800,'월드콘':1500,'메로나':1000}
ice=list(icecream.keys())
print(ice)
#%%
icecream={'탱크보이':1200,'폴라포':1200,'빵빠레':1800,'월드콘':1500,'메로나':1000}
price=list(icecream.values())
print(price)
#%%
def d(n):
    for i in str(n):
        n+=int(i)
    return n
#%%


a=set((range(1,5001)))
b=set()
for i in a:
    b.add(d(i))
print(sum(a-b))
#%%
print(fib_list(10))
print(d(10))
#%%
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
#%%
def d(n):
    num_list=[]
    for i in range(1,n+1):
        if n%i==0:
            num_list.append(i)
    return num_list

print(d(10))
#%% 함수안에 함수 
def two(n) :
    fib_list_first = fib_list(n)
    result_list = []
    for temp_elem in fib_list_first:
        temp_data = d(temp_elem)
        result_list.append(temp_data)
    print(result_list)    
    return result_list
#%%
def two(n):
    fib_list_first=fib_list(n)
    result_list=[]
    for temp_elem in fib_list_first:
        temp_data=d(temp_elem)  #d(temp_element)=각 항목의 약수 구하기
        result_list.append(temp_data)
    print(result_list)
    return result_list

two(10)
#%%
class Foo:
  pass
obj = Foo()
obj.foo = 2
print(obj.foo) 
#%% 8 counting
total=0
for i in range(1,10001):
    for n in str(i):
        if n=='8' : total+=1
print(total) 
#%% yayy!!!
total=0
for i in range(1,10001):
    for n in str(i):
        if n=='8':
            total+=1
print(total)        
#%%
import requests 
import json
import pandas as pd
import re

url = 'https://api.nasdaq.com/api/calendar/splits'

headers = {
        'referer': 'https://www.nasdaq.com/',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36'
        }
params = {
    'date': '2021-01-21'}


#%%

response = requests.get(url, headers = headers, params = params)

#%%
print(type(response.text))
print(type(data_df.executionDate))
#%%
text = json.loads(response.text)

#%%
print(text['data'])

#%%

print(text['data']['rows'])



#%%

print(type(text['data']['rows']))
#%% 표표표표표표표표표표표푶표표표표표표표표표표푶

data_df = pd.DataFrame(text['data']['rows'])
print(data_df)
#%% TQQQ 조건에 맞는 표 출력
print(data_df.loc[1, :])

#%%
print(response.text['data'])


#%% 
#숙제
#symbol의 첫 글자가 A,B,C,D,E,F,G 인것을 가져오세요 
data=data_df['symbol'].str.startswith(("A" , "B" , "C" , "D" , "E" , "F" ,"G"))
#%%
print(data_df)
#%%
print(data_df['payableDate'])

#%%
print(data_df['payableDate'][:2])
#%% 
print(data_df['ratio'].str.startswith('1 : '))

#%%
temp_data = data_df.loc[26,:]
#%%
print(temp_data['ratio'].find(':'))

point = temp_data['ratio'].find(':')
#%% 1:3이상인 것만 가져오기 

index_list=list(data_df.index)  
result_list=[]

for temp_index in index_list:
    print(temp_index)
    temp_data = data_df.loc[temp_index, :]
    print(temp_data['ratio'])
    if temp_data['ratio'].find(':') == -1:
        pass
    
    else:
        point = temp_data['ratio'].find(':')
        if float(temp_data['ratio'][point+1:])/float(temp_data['ratio'][:point])>1:
            print(data_df.loc[temp_index, :])
            
            result_list.append(temp_index)
            1
        else:
            pass
        
print(result_list)
#%%
index_list=list(data_df.index)  
result_list=[]

for temp_index in index_list:
    print(temp_index)
#%%
print(data_df.loc[result_list, :])


#%%
print('333:33'.find(':'))
print('333:33'[:3])
print('333:33'[3+1:])
#%%
def d(n):
    num_list=[]
    for i in range(1,n+1):
        if n%i==0:
            num_list.append(i)
    return num_list

print(d(10))

#%%  index로 column접근하기

temp_data = data_df.loc[temp_index, :]
print(temp_data['ratio'])
if temp_data['ratio'].find(':') == -1:
    
    
    
#%%

if data_df.loc[temp_index,:]['ratio'].find(':') == -1:

#%% ratio 데이터로 가져오기

print(data_df.loc[result_list, :])

#%% 날짜차이가 2이상인 것 가져오기

payable_list=list(data_df.payableDate)
exe_list=list(data_df.executionDate)

#%%
print(payable_list)
#%%
for temp_payable in payable_list:
    temp_p=data_df.loc[temp_payable, :]
    print(temp_p)

#%% ratio 

index_list=list(data_df.index)  
result_list=[]

for temp_index in index_list:
    
    print(temp_index)
    temp_data = data_df.loc[temp_index, :]
    print(temp_data['ratio'])
    
    if temp_data['ratio'].find(':') == -1:
        pass
    
    else:
        point = temp_data['ratio'].find(':')
        if float(temp_data['ratio'][point+1:])/float(temp_data['ratio'][:point])>1:
            print(data_df.loc[temp_index, :])
            
            result_list.append(temp_index)
        else:
            pass
        
print(result_list)
#%%
print(data_df.loc[result_list, :])
#%%

temp_data = data_df.loc[1,:]
if temp_data['payableDate'] :
    
print(data_df.loc[1,:]['payableDate'])
#%%

index_list=list(data_df.index)  
result_list=[]

for temp_index in index_list:
    print(temp_index)
    temp_data = data_df.loc[temp_index, :]

    if abs(int(temp_data['payableDate'][3:5])-int(temp_data['executionDate'][3:5]))>=2:
        print(data_df.loc[temp_index, :])
        
        result_list.append(temp_index)
        
    else:
        pass
    
print(result_list)

#%%
print(data_df.loc[result_list, :])        
#%%

index_list=list(data_df.index)
result_list=[]

for temp_index in index_list:
    print(temp_index)
    temp_data=data_df.loc[temp_index, :]
    print(temp_data['name'])
    
    c=temp_data['name']
    if c.count(' ')>=3:
        print(data_df.loc[temp_index, :])
        result_list.append(temp_index)
        
    else:
        pass
    
print(result_list)
#%%
print(data_df.loc[result_list, :])
#%%class

class Jalynne:
    
    def __init__(self, name, sex, dislike):
        self.name=name
        self.sex=sex
        self.dislike=dislike

    def info(self):
        print('제 이름은', self.name , '입니다')
        print('저는', self.sex , '입니다')
        print('저는', self.dislike ,'을 싫어합니다')
        
#%%
jn = Jalynne('Jalynne',
    'female',
    'salmon'
    )

#%%
jn.info()
#%% 진심 울고싶음
숙제

1. Jalynne 클래스를 만든다 .

2. init에서 다음의 변수를 지정한다.
 - 1. Jalynne의 나이, 성별
 - 2. Jalynne이 점심 식사 후에 간식으로 먹고 싶은 것을 List 의 형태로 저장
 - 3. 2월 15일에서 27일까지의 평일을 list로 저장

3. 몇가지 메소드를 만들어 본다 ( 클래스 안에서 정의된 함수를 메소드라고 함 )
 - 1. day_pick 이라는 메소드를 만든다.
 - 메소드의 역할 : 2월 15일에서 27일 까지의 날짜 ( init에서 저장한 List) 를 받아서 하루를  랜덤으로 선택해서 return으로 주는 메소드.

 - 2. dessert_pick 이라는 메소드를 만든다.
 - 메소드의 역할 : 간식으로 먹고싶은 것을 List의 형태로 저장한 것을 받아서 그중 한가지를  랜덤으로 선택해서 return으로 주는 메소드 

 - 3. dessert_result 이라는 메소드를 만든다
 - 메소드의 역할 : 메소드 안에서 day_pick, dessert_pick메소드를 사용해서 2월 15일에서  27일중 평일의 날짜 하나를 return하고, 또 간식으로 먹고싶은 것 하나를 return 하는 메소드.

 - 이때 return은 두가지 값을 한번에 return해야 함.
#%%

import random
import datetime

#%%
print(datetime.datetime.today())
print(datetime.datetime.now())
#%%

t = ['월','화','수','목','금','토','일']
temp_time = datetime.datetime.now() + datetime.timedelta(days = 1)
print(temp_time)
print(t[temp_time.weekday()])
#%%
        self.age=age
        self.sex=sex
        age,sex,
#%% 

import random
import datetime

class Jalynne:
    def __init__(self,snack_list,weekday_list):
        self.snack=snack_list
        self.weekday=weekday_list
        
    t = ['월','화','수','목','금','토','일']
    temp_time = datetime.datetime.now() + datetime.timedelta(days = 1)    
    def day_pick(self): 
        for i in range():                             
           if t[temp_time.weekday()]='토' or '일'     
               pass
           else:
               print()
           
    def dessert_pick(self):
        s=self.snack
        print(random.choice(s))
        
    def dessert_result(self):
        self.dessrt_pick
        
#%%
from datetime import date,timedelta
d1=date(2021,2,15)
d2=date(2021,2,27)
delta=d2-d1
for i in range(delta.days+1):
    print(d1+timedelta(days=i))
#%%
import random
from datetime import date,timedelta

class Jalynne:
    def __init__(self,snack):
        #self.weekday=weekday
        self.snack=snack

    def day_pick(self): 
        t = ['월','화','수','목','금','토','일']
        temp_time = datetime.datetime.now() + datetime.timedelta(days = 1)    
        d1=date(2021,2,15)
        d2=date(2021,2,27)
        delta=d2-d1

        day_list=[]
        for i in range(delta.days+1):
           if t[(d1+timedelta(days=i)).weekday()]=='토' or t[(d1+timedelta(days=i)).weekday()]=='일' : 
               pass
           else:
               day_list.append(d1+timedelta(days=i))
               
        target_date = random.sample(day_list,1)
        return target_date
        
    def dessert_pick(self):
        s=self.snack
        print(random.sample(s,1))
        
    def dessert_result(self):
        self.day_pick       
        print(target_date)   
        
    
#%%

s=Jalynne(['a','b','c','d','e'])
s.dessert_pick()

#%%

d=Jalynne()
d.day_pick()

#%%
import random
from datetime import date,timedelta

class Jalynne:
    def __init__(self,snack):
        #self.weekday=weekday
        self.snack=snack

    def day_pick(self): 
        t = ['월','화','수','목','금','토','일']
        temp_time = datetime.datetime.now() + datetime.timedelta(days = 1)    
        d1=date(2021,2,15)
        d2=date(2021,2,27)
        delta=d2-d1
        day_list=[]
        for i in range(delta.days+1):
           if t[(d1+timedelta(days=i)).weekday()]=='토' or t[(d1+timedelta(days=i)).weekday()]=='일' : 
               pass
           else:
               day_list.append(d1+timedelta(days=i))
        return day_list
        
    def dessert_pick(self):
        s=self.snack
        print(random.sample(s,1))
        
    def dessert_result(self):
        day_list=self.day_pick()
        # print(day_list)
        print(random.sample(day_list,1))
         
#%%
s=Jalynne(['a','b','c','d','e'])
s.dessert_result()

#%% 완성

import random
from datetime import date,timedelta

class Jalynne:
    def __init__(self,sdate,edate,snack):
        self.sdate=sdate
        self.edate=edate
        self.snack=snack
    
    #랜덤으로 날짜 고르기
    def day_pick(self): 
        t = ['월','화','수','목','금','토','일']
        temp_time = datetime.datetime.now() + datetime.timedelta(days = 1)    
        delta=self.edate-self.sdate
        day_list=[]
        for i in range(delta.days+1):
           if t[(self.sdate+timedelta(days=i)).weekday()]=='토' or t[(self.sdate+timedelta(days=i)).weekday()]=='일' : 
               pass
           else:
               day_list.append(self.sdate+timedelta(days=i))
        print((random.sample(day_list,1)))
    
    #랜덤으로 디저트 고르기    
    def dessert_pick(self):
        s=self.snack
        print((random.sample(s,1)))
    
    #     
    def dessert_result(self):
        day_list=self.day_pick()
        s=self.dessert_pick()
    
        return day_list,s
       
#%%
s=Jalynne(date(2021,2,15),
    date(2021,2,27),
    ('greentea_frappuccino_with_java_chip_without_whipped_cream','Thank_you','James')
    )

s.dessert_result()
#%% Jalynne_exercise3 
import pandas as pd
df=pd.read_excel(r'/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/Jalynne_excercise3.xlsx', engine = 'openpyxl')
print(df) 

#%%
print(df.loc[3, :])

#%% runcell204 
import pandas as pd
corp_df=pd.read_excel('C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/corp_info.xlsx', engine = 'openpyxl')
print(corp_df)
#%%
print(corp_df)

#%%
corp_df['stock_code'].dropna()
#%% 0번째 행의 정보 출력하기
temp_data=df.loc[0,:]
print(temp_data.keys)
#%%
print(temp_data.values[1])
#%%
print(temp_data['총장'])

#%% stock_code중에서 빈 셀 제외한 dataframe 가져오기
i=corp_df.dropna(subset=['stock_code'])
print(i)

#%% stock_name 가져오기 완성
index_list=list(i.index)
result_list=[]

for temp_list in index_list:
    temp_data=i.loc[temp_list,:]
    p=temp_data['stock_name']
    result_list.append(p)
print(result_list)
#%% result_list의 형태는 list
print(type(result_list))
#%% 모든 index의 값 (1)
for i in df.index:
    temp_data=df.loc[i,:]
    print(temp_data.keys)

#%%
print(temp_data)

#%% key값으로 value값 출력하기 (2) 데이터 값 하나하나
import numpy as np

values=[]
for key in temp_data.keys():
    if str(temp_data[key]) != 'nan':
            values.append(temp_data[key])
print(values)

#%%
print(temp_data)
#%%
index_list=list(df.index) #Jalynne_exercise3 df의 index_list
#%%
print(result_list)
#%%
print(values)
#%%
print(index_list[:10000])

#%% 
from tqdm import tqdm
#%% 

for temp_index in tqdm(index_list):
    temp_data = df.loc[temp_index, :]
    values = list(temp_data.values)
    final=[]
    flag = 0
    for a in result_list:
        for b in values:
            if a==b:    
                final.append(index_list)
                flag = 1
                break
            else:
                pass
        if flag == 1:
            break
        else:
            pass
#%% RE
for temp_index in tqdm(index_list[:10000]):
    temp_data = df.loc[temp_index, :]
    values = list(temp_data.values)
    final=[]
    flag = 0
    for a in result_list:
        for b in values:
            b=b.replace(' ','')       
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

#%% Rere try except

for temp_index in tqdm(index_list[:50000]):
    temp_data = df.loc[temp_index, :]
    values = list(temp_data.values)
    final=[]
    flag = 0
    for a in result_list:   
        for b in values:  
            try: 
                b=b.replace(' ','')       
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
#%%
print(len(index_list))
#%%
print(index_list[:10000])epdl
#%%

print(index_list)
#%%
print(final)
#%%
print(result_list.index('LG유플러스'))
print(result_list[1167])

#%% True
print('총장' in temp_data.keys())
#%% 
#1. 국적 찾는 함수

def country(nation,df):
      
    p=df.dropna(subset=['정당'])
    index_listp=list(p.index)
    p=p.loc[index_listp,:]
    p=p.dropna(axis=1,how='all')
    n=p.dropna(subset=['국적'])


    index_list=list(n.index)
    result_list=[]
    
    for temp_index in index_list:
        temp_data=n.loc[temp_index,:]
        
        if temp_data['국적']==nation: 
            result_list.append(temp_index)
        else:
            pass
    
    return n.loc[result_list,:]

print(country('대한민국',df))
#%%
#2. 출생일 특정년도 이상(?) 찾고+ 생년월일 yyyymmdd로 바꾸는 함수


def birth(year,df):
    
#     p=df.dropna(subset=['정당'])
    p=df.dropna(subset=['출생일'])
    index_listp=list(p.index)
    p=p.loc[index_listp,:]
    p=p.dropna(axis=1,how='all')
    p=p.dropna(axis=1,thresh=800)
    
    birth_year=p['출생일'].astype(str).str[:4]
    birth_year=birth_year[birth_year.str.isdecimal()]
    birth_year=birth_year.astype(int)
    birth_year=birth_year[birth_year>=year]
    index_list1=birth_year.index.tolist() #index_list1=특정년도 이상(>=)에 해당하는 인덱스 리스트

    
    bb=p.loc[index_list1,:]
    index_list1=list(bb.index)
    result_list1=[]
    result_list11 = []
    for temp_index1 in index_list1:
        temp_data1=bb.loc[temp_index1,:]
    
        point=temp_data1['출생일'].find('(')
        point1=temp_data1['출생일'].find(')')
        target = temp_data1['출생일'][point+1:point1].strip().replace(' ','')
        
        if bool(re.match('.*-.*-.*',temp_data1['출생일'][point+1:point1].strip()))== True:
            z = temp_data1['출생일'][point+1:point1].replace('-','') 
            result_list1.append(temp_index1)
            result_list11.append(z)
        else:
            result_list11.append('no result')

    
    bb['출생일'] = result_list11
    return bb.loc[result_list1,:]

print(birth(1930,df)) 
#%%
#3. 이름 한글만 뽑아내기

def names(name,df):
    p=df.dropna(subset=['출생일'])
    index_listp=list(p.index)
    p=p.loc[index_listp,:]
    p=p.dropna(axis=1,how='all')
    p=p.dropna(axis=1,thresh=800)

    result_list2=[]
    index_list2=list(p.index)

    for temp_index2 in index_list2:
        
        if name=='한글':
        
            temp_data2=p.loc[temp_index2,:].copy()
            temp_data2['_id']=str( temp_data2['_id'])
            hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
            h=hangul.sub('',temp_data2['_id'])
            result_list2.append(h)          #result_list2에 h를 append하는것=temp_data2의 _id중에 한글문자만 append

        elif name=='영어':
            
            temp_data2=p.loc[temp_index2,:].copy()
            temp_data2['_id']=str( temp_data2['_id'])
            eng = re.compile('[^a-zA-Z0-9]')
            e=eng.sub('',temp_data2['_id'])
            result_list2.append(e)

        else:
            pass
    
    p['_id'] = result_list2              #그렇게 append해서 만든 list를 p['_id']로 치환하기
    return p
print(names('한글',df))
#%%
#4.  종교 예쁘게 만들기 완성

hangul = re.compile('[^ ㄱ-ㅣ가-힣  ]+')

def religion(df):
    p=df.dropna(subset=['정당'])
    index_listp=list(p.index)
    p=p.loc[index_listp,:]
    p=p.dropna(axis=1,how='all')
    p=p.dropna(axis=1,thresh=800)
    
    #for1
    index_list3=list(p.index)
    result_list = []
    for temp_index3 in index_list3:
        try:
            temp_data3=p.loc[temp_index3,:]
            r=temp_data3['종교'].replace("[","(").replace("]",")").replace("{","(").replace("}",")").replace(" ","")
            r=re.sub(r'\([^)]*\)', '', r).replace(')','').replace('(','')
            result_list.append(r) 
        except:
            result_list.append('no result')
    
    p['종교']=result_list
    
    #for2
    result_list1=[]
    for temp_result in result_list:
        if bool(re.match('.*→.*',temp_result)) == True:
            point = [i.start() for i in re.finditer('→',temp_result)]
            result_list1.append(temp_result[point[-1]+1:].strip())
        else:
            result_list1.append(temp_result)
    
    p['종교']=result_list1
    
    #for3
    result_list2=[]
    for r in result_list1:
        if "개신교" in r:
            r=r.replace(r,"기독교")
            result_list2.append(r)
            
        elif "성공회" in r:
            r=r.replace(r,"성공회")
            result_list2.append(r)
            
        elif "카톨릭" in r:
            r=r.replace(r,"카톨릭")
            result_list2.append(r)            
        
        elif "가톨릭" in r:    
            r=r.replace(r,"카톨릭")
            result_list2.append(r)    

        elif "무교" in r:    
            r=r.replace(r,"무교")
            result_list2.append(r)                

        elif "무종교" in r:    
            r=r.replace(r,"무교")
            result_list2.append(r)   
            
        elif "무신론" in r:    
            r=r.replace(r,"무교")
            result_list2.append(r)                   

        elif "이슬람" in r:    
            r=r.replace(r,"이슬람교")
            result_list2.append(r)             
        
        elif "정교" in r:    
            r=r.replace(r,"정교회")
            result_list2.append(r)  
        
        elif "없음" in r:    
            r=r.replace(r,"no result")
            result_list2.append(r)                          
            
        else:
            result_list2.append(r)
     
    p['종교']=result_list2
    
    return p

print(list(religion(df)['종교']))
#%%
#5. 출생일이 있는 데이터 중 정당을 예쁘게 찾고 학력 정리하기

import re

def party(df):
    p=df.dropna(subset=['출생일'])
    index_listp=list(p.index)
    p=p.loc[index_listp,:]
    p=p.dropna(axis=1,how='all')
    p=p.dropna(axis=1,thresh=800)
    
    #for1
    index_list3=list(p.index)
    result_list = []
    for temp_index3 in index_list3:
        try:
            temp_data3=p.loc[temp_index3,:]
            r=temp_data3['정당'].replace("[","(").replace("]",")").replace("{","(").replace("}",")").replace(" ","")
            r=re.sub(r'\([^)]*\)', '', r).replace(')','').replace('(','')
            result_list.append(r) 
        except:
            result_list.append(r)
    
    p['정당']=result_list
    
    #for 2
    result_list1=[]
    for temp_result in result_list:
        if bool(re.match('.*→.*',temp_result)) == True:
            point = [i.start() for i in re.finditer('→',temp_result)]
            result_list1.append(temp_result[point[-1]+1:].strip())
        else:
            result_list1.append(temp_result)
    
    p['정당']=result_list1    
    
    #for 3 특수문자 없애기
    result_list2=[]
    for s in result_list1:
        try:
            s=re.sub(r'[^ ㄱ-ㅣ가-힣A-Za-z]', '', s) 
            result_list2.append(s)
        except:
            result_list2.append(s)
    p['정당']=result_list2    
    
    
    #for 4 학력 빈칸 다 없앰
    result_list4 = []
    for temp_index3 in index_list3:
        try:
            temp_data3=p.loc[temp_index3,:]
            s=temp_data3['학력'].replace(" ","")
            result_list4.append(s) 
        except:
            result_list4.append(temp_data3['학력'])
    
    p['학력']=result_list4

   #for 5 univ 이름 append하기
    index_list=list(univ_df.index)
    univ_list=[]
    for temp_index in index_list:
        temp_data=univ_df.loc[temp_index,:]
        pp=temp_data['_id']
        univ_list.append(pp)
    
  
    final=[]
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

    #print(len(final))
    #print(len(p['학력']))
    
    p['학력']=final                 
    print(final)
    
    return p 

    print(list(index(p['학력'])))
     
print(party(df))
#%%
    remove_text ='유교(성리학)'
    print(re.sub(r'\([^)]*\)', '', remove_text))
#%%
    
    #for 4 학력 빈칸 다 없앰
    result_list4 = []
    for temp_index3 in index_list3:
        try:
            temp_data3=p.loc[temp_index3,:]
            s=temp_data3['학력'].replace(" ","")
            result_list4.append(s) 
        except:
            result_list4.append(temp_data3['학력'])
    
    p['학력']=result_list4

   #for 5 univ 이름 append하기
    index_list=list(univ_df.index)
    univ_list=[]
    for temp_index in index_list:
        temp_data=univ_df.loc[temp_index,:]
        pp=temp_data['_id']
        univ_list.append(pp)
    
  
    final=[]
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

    #print(len(final))
    #print(len(p['학력']))
    
    p['학력']=final                 
    print(final)
    
    return p 

    print(list(index(p['학력'])))
     
print(party(df))
#%%
import pandas as pd
import re
df=pd.read_excel(r'/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/Jalynne_excercise3.xlsx', engine = 'openpyxl')
#%% social event 함수로 만들기

import re
import numpy as np
import pandas as pd
from tqdm import tqdm

def social_event(df):
    df=df.dropna(axis=1,how='all')
    df_list=list(df.index) 
    events=['대선','올림픽','월드컵','아시안 게임']
    
    temp_list=[]
    for b in tqdm(df.index[:10000]):
        temp_data=df.loc[b,:]    
        
        values=[]   
        for key in temp_data.keys():
            if str(temp_data[key]) != 'nan':
                values.append(temp_data[key])
        flag=0            
        for e in events:
            for v in values:
                try :
                    v=v.replace(' ','')
                    if e in v:
                        temp_list.append(b)
                        flag=1
                        break
                    else:
                        pass
                except:
                    pass
            if flag==1:
                break
            else:
                pass
    df=df.loc[temp_list,:]
    print(list(df.index))
    
    df=df.query('`개최 도시`.notnull() or 참가국.notnull() or 경력.notnull() or `참가 선수`.notnull() or 대회.notnull()')
    return df

print(social_event(df))
#%%
    #콤마 뒤의 것만 가져온 다음에 이어붙이기!!!!!
    
    result_list8=[]
    for m in list(p['창당']):
        
        m=m.replace(' ','')
        point=m.find('년')
        point1=m.find('월')
        point2=m.find('일')
        
        #mm월 
        try:
            if len(m[point+1:point1]) == 1:
                m=m[:point+1]+'0'+m[point1-1:]
                result_list8.append(m)
            else:
                result_list8.append(m)
        except:
            pass
    p['창당']=result_list8
        
    result_list9=[]
    for mm in list(p['창당']):
        
        mm=mm.replace(' ','')
        point=mm.find('년')
        point1=mm.find('월')
        point2=mm.find('일')
        
        #mm월 
        try:
            if len(mm[point1+1:point2]) == 1:
                mm=mm[:point1+1]+'0'+mm[point2-1:]
                result_list9.append(mm)
            else:
                result_list9.append(mm)
        except:
            pass
    p['창당']=result_list9
    
    #년월일 없애기
    result_list10 = []
    for mmm in list(p['창당']):
        try:
            mmm=mmm.replace("년","").replace("월","").replace("-","").replace("일","").replace("\n","")
        except:
            result_list10.append(mmm)
    
    p['창당']=result_list10        

#%%
    result_list8 = []
    for temp_elem in list(p['창당']):
        try:
            if temp_elem.find('년') == -1: # '년'이 없으면
                result_list8.append('no result') # no result
                continue
            else: #'년'이 있으면
                temp_elem = temp_elem.replace(' ','').strip() #'년'이 있으면 공백 없애주기
                count = [i.start() for i in re.finditer('년',temp_elem)] #뒤에서 '년'찾아준걸 count로 지정
                temp_elem = temp_elem[max(0,count[len(count)-1]-5):count[len(count)-1]+8] 
                # temp_elem = temp_elem.replace(' ','').strip() 
                hangul_numb = re.compile('[^가-힣 0-9]+') #정규표현식 한글만
                temp_elem = hangul_numb.sub('',temp_elem) #temp_elem에서 한글빼고 다 없애주기
               
                str1_, str2_, str3_ = '', '', '' # '' 형태를 str1_ str2_, str3_로 지정해주기  
                point1 = temp_elem.find('년') # temp_elem에서 '년'을 찾은 것을 point1
                str1_ = temp_elem[:point1] #'년' 앞에까지 가져온것=str1 = yyyy가 되겠지 아마도
                if temp_elem.find('월') != -1: #if '월'이 있으면
                    point2 = temp_elem.find('월') #temp_elem 에서 '월'을 찾은것 = point2 
                    str2_ = temp_elem[point1+1:point2] #'년' 뒤에서부터 월까지=str2 는 mm이 되겠지 아마도
                    if temp_elem.find('일') != -1: #'월'이 있는 것 중에 if'일'이 있으면  
                        point3 = temp_elem.find('일') #temp_elem 에서 '월'을 찾은것 = point3 
                        str3_ = temp_elem[point2+1 : point3] #'월' 뒤에서부터 '일'까지=str3 는 dd가 되겠지 아마도
                    else: #'일'이 없으면 pass
                        pass
                else: #'월'이 없으면 pass
                    pass
               
                if len(str2_) == 0: #if mm이 없으면
                    result_list8.append(str1_) # yyyy만 append
                    continue
                elif len(str2_) == 1: #else if mm이 한자리면 0추가해주기
                    str2_ = '0' + str2_
                else: #mm이 있으면 pass
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
#%%
    
    #[폐막식] mm월로 만들기
    result_list=[]
    for aa in list(df['폐막식']):
        
        try:
            aa=aa.replace(' ','')
            
            point=aa.find('년') 
            point1=aa.find('월')
            point2=aa.find('일')
        
            # O월을 OO월로 바꾸기
            try:
                if len(aa[point+1:point1])==1:
                    aa=aa[:point+1]+'0'+aa[point1-1]
                    result_list.append(aa)
                else:
                    result_list.append(aa)
            except:
                pass
        
        except:
            result_list.append('no result')
    
    df['폐막식']=result_list
    
    #[폐막식] dd일로 만들기
    result_list1=[]
    for bb in list(df['폐막식']):
        
        try:
            bb=bb.replace(' ','')
            point=bb.find('년')
            point1=bb.find('월')
            point2=bb.find('일')        
        
            try:
                if len(bb[point1+1:point2]) == 1:
                    bb=bb[:point1+1]+'0'+bb[point2-1:]
                    result_list1.append(bb)
                else:
                    result_list1.append(bb)
            except:
                pass
        
        except:
            result_list1.append('no result')
            
    df['폐막식']=result_list1
    
    #[폐막식] 년월일 없애기
    result_list2 = []
    for cc in list(df['폐막식']):
        try:
            cc=cc.replace("년","").replace("월","").replace("일","")
            result_list2.append(cc)
        except:
            result_list2.append(cc)
    
    df['폐막식']=result_list2        

#%%
    #[폐막식] mm월 dd일로 만들기
    result_list=[]
    for aa in list(df['폐막식']):
        
        try:
            aa=aa.replace(' ','')
            
            point=aa.find('년') 
            point1=aa.find('월')
            point2=aa.find('일')
        
            try:
                if len(aa[point+1:point1])==1: 
                    aa=aa[:point+1]+'0'+aa[point1-1:]
                    
                    if len(aa[point1+2:point2+1]) == 1: #월이 한자리 이면서 일이 한자리이면
                        aa=aa[:point1+2]+'0'+aa[point2:]
                        result_list.append(aa)
                        
                    else: #월이 한자리이면서 일이 두자리이면
                        result_list.append(aa)
                
                else: #월이 두자리이면
                    aa=aa[:point1+1]+'0'+aa[point2-1:]
                    result_list.append(aa)
            except:
                result_list.append(aa)
        except:
            result_list.append(aa)
    
    df['폐막식']=result_list
    
    #괄호 없애고 년월일 없애기
    result_list1=[]
    for aaa in list(df['폐막식']):
        try:
            aaa=re.sub(r'\([^)]*\)', '',aaa)
            aaa=aaa.replace("개최되지않음","").replace("년","").replace("월","").replace("일","")
            result_list1.append(aaa)
        except:
            result_list1.append(aaa)
    df['폐막식']=result_list1