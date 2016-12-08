import random
import sets

num_lines = sum(1 for line in open('training.txt'))

f = open('training.txt', 'r')
trainf = open('validation.txt','w')
testf = open('train.txt','w')
s = sets.Set([])

while len(s) < 50:
	i = random.randint(0, num_lines)
	if i not in s:
		s.add(i)

counter = 0
for line in f:
	if counter in s:
		trainf.write(line)
	else:
		testf.write(line)
	counter += 1