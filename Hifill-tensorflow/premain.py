from LoFTR.loftr import loftrGenerate
from LoFTR.findJpg import returnPair, checkDirectory, getDirs
from LoFTR.util import returnBaseName
from MyVersion.util import inference
import random

dir = input("Please input file dir: ")
pair = returnPair(dir)

blocksXY = [0,0,0,0]

blocksXY[0] = int(input("x1: "))
blocksXY[1] = int(input("x2: "))
blocksXY[2] = int(input("y1: "))
blocksXY[3] = int(input("y2: "))
loftrGenerate(pair[0][0], pair[0][1], dir ,blocksXY)

inference(dir)