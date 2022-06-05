from os import listdir
from os.path import isfile, join

mypath = "/home/ltoschi/Documents/hackatagro/Soil types - selected/Black soil"
onlyfiles = [(mypath + "/" + f) for f in listdir(mypath) if isfile(join(mypath, f))]

for file in onlyfiles:
    print(file)