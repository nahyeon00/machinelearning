from google.colab import files
uploaded = files.upload()


with open('news.txt', 'r',encoding='utf-8') as f:
    line = f.read() #type str
# 문자열 나누기(split)
    #split() 괄호안에 아무것도 넣어주지 않으면 공백 기준으로 문자열 나누어 list로 저장
str = line.split()
dic = {}

for i in str:
    dic[i] = dic[i]+1 if i in dic else 1

print('전체 단어 수:', len(dic))

# 1) 토큰화된 단어들의 빈도수를 출력
print('1) 토큰화된 단어들의 빈도수를 출력')
print(dic)

#dic.items() 호출 시 dict의 key,value를 tuple로 묶어 list로 반환해줌
dic.items()

#sorted(정렬할 데이터, key 파라미터, reverse 파라미터)
    #첫 번째 매개변수로 들어온 데이터를 새로운 정렬된 리스트로 만들어서 반환해 주는 함수
    #key 파라미터는 어떤 것을 기준으로 정렬할 것인가에 대한 기준
    #reverse=False 로 오름차순 정렬이 기준
#lambda(인자 : 표현식): 익명함수
sort_dict=sorted(dic.items(), key=lambda x: x[1], reverse=True)

# 2) 많이 나온 단어 순으로 출력
print('\n2) 많이 나온 단어 순으로 출력')
print(sort_dict)

# 3) 많이 나온 단어 5개를 출력
print('\n3) 많이 나온 단어 5개를 출력')
print(sort_dict[:5])

# 4) 30번 이상 나온 단어들을 출력
print('\n4) 30번 이상 나온 단어들을 출력')

for key, value in sort_dict:
    if value >= 30:
        print(key, ':', value)
