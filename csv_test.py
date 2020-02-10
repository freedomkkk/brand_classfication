import csv
outputFile = open('output.csv', 'w', newline='')
outputWriter = csv.writer(outputFile,delimiter=' ')
a='ss'
b='kk'
outputWriter.writerow([a,b])
outputWriter.writerow(['Hello, world!', 'eggs', 'bacon', 'ham'])
outputWriter.writerow([1, 2, 3.141592, 4])
outputFile.close()

20   50