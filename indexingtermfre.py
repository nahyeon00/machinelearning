import collections

with open('news.txt', 'r', encoding='utf-8') as f:
    line = f.read().lower()
# 문자열 나누기(split)
    #split() 괄호안에 아무것도 넣어주지 않으면 공백 기준으로 문자열 나누어 list로 저장
token = line.split()
token2idx = collections.defaultdict(lambda: -1)

top = collections.Counter(token).most_common(100)

for word in top:
    token2idx[word] = len(token2idx)

print(token2idx)