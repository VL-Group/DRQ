from random import shuffle

with open('./data/imagenet/database.txt', 'r') as fp:
    database = fp.readlines()


i = -1
index = 0

count = []

result = []

for d in database:
    cl = int(d.split()[1])
    if cl != i:
        i = cl
        if len(count) > 0:
            shuffle(count)
            result += count[:100]
            count.clear()
    count.append(d)


if len(count) > 0:
    shuffle(count)
    result += count[:100]
    count.clear()

assert len(result) == 10000

shuffle(result)

with open('./data/imagenet/train.txt', 'w') as fp:
    fp.writelines(result)