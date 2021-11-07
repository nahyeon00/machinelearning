import collections

with open('news.txt', 'r',encoding='utf-8') as f:
    line = f.read() #type str
token = line.split()
token2idx = collections.defaultdict(lambda: -1)

for word in token:
    if word not in token2idx:
        token2idx[word] = len(token2idx)

print(token2idx)
