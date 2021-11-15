text1 = "Json is an open standard file format"
text2 = "xml is another open standard file format"


def ngram(text, n, pad_left=False, pad_right=False):
    list = []
    k = []
    for i in text:
        k.append(i)
    if pad_left == True:
        k.insert(0, '<s>')
    if pad_right == True:
        k.append('</s>')

    for i in range(len(k)-n+1):
        tmp = []
        for j in range(n):
            tmp.append(k[i+j])
        list.append(tmp)
    return list


def ngram_overlap(one, two, a):
    one = one.split()
    two = two.split()
    first = ngram(one, a)
    second = ngram(two, a)
    result = [x for x in first if x in second]
    return result


common = ngram_overlap(text1, text2, 2)
#같이 나타난 n-gram 출력
print(common)