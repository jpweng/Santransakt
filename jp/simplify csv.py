
#%%
import csv
import os

main_folder = r"D:\dev\SanTransakt"
data_folder = os.path.join(main_folder, 'data')

def CsvSimplification (fileNameIn, fileNameOut, maxRows):
    file = open (fileNameIn)
    reader = csv.reader (file)
    rows = []
    curRow = 0
    for row in reader:
        rows.append(row)
        curRow += 1
        if curRow >= maxRows:
            break

    file.close ()

    simple_file = open (fileNameOut, mode='w')
    simple_csv = csv.writer (simple_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')

    for row in rows:
        simple_csv.writerow (row)

    simple_file.close ()

trainIn = os.path.join(data_folder,"train.csv")
trainOut = os.path.join(data_folder, "train_simpified.csv")

CsvSimplification (trainIn, trainOut, 10000)

testIn = os.path.join(data_folder,"test.csv")
testOut = os.path.join(data_folder, "test_simpified.csv")

CsvSimplification (testIn, testOut, 2000)
