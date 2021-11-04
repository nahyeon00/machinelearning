import collections

with open('news.txt', 'r',encoding='utf-8') as f:
    line = f.read()#type str

# 문자열 나누기(split)
    #split() 괄호안에 아무것도 넣어주지 않으면 공백 기준으로 문자열 나누어 list로 저장
str = line.split()
#counter: 각 요소의 개수를 세고 많은 것 부터 dictionary 형태로 반환
cnt = collections.Counter(str)

# 1) 많이 나온 단어 순으로 출력
print('\n1) 많이 나온 단어 순으로 출력')
print(cnt)


# 2) 많이 나온 단어 5개를 출력
print('\n2) 많이 나온 단어 5개를 출력')
print(cnt.most_common(5)) #리스트 속 튜플형식

# 3) 30번 이상 나온 단어들을 출력
print('\n3) 30번 이상 나온 단어들을 출력')
all = cnt.most_common(len(cnt))

for key, value in all:
    if value >= 30:
        print(key, ':', value)
