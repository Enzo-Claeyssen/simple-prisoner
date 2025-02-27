from ipd import *
from strategies import *
from tqdm import tqdm
from QLearning import QLearning

bag=[Periodic("D"), HardMajority(), Tft(), Spiteful(),  Gradual()]

for _ in tqdm(range(100000), desc = "Training...") :
    m = Meeting(g, QLearning(True), Tft(), length=10)
    m.run()

m = Meeting(g, QLearning(False), Tft(), length=10)
m.run()
m.prettyPrint()

QLearning.exportQTable()
"""
t= Tournament(g,bag)        # default: length=1000
e= Ecological(t)            # default: pop=100
e.run()
e.tournament.matrix
e.historic
e.drawPlot()
"""