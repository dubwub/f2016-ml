import random
import sets

num_lines = sum(1 for line in open('combined.txt'))

f = open('combined.txt', 'r')
trainf = open('training.txt','w')
testf = open('test.txt','w')
s = sets.Set([])

while len(s) < 300:
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