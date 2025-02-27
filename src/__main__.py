from ipd import *
from strategies import *
from tqdm import tqdm
from QLearning import QLearning

bagTrain=[Periodic("D"), HardMajority(), Tft(), Spiteful(),  Gradual(), QLearning(True)]
bag = [Periodic("D"), HardMajority(), Tft(), Spiteful(),  Gradual(), QLearning(False)]

for _ in tqdm(range(1000), desc = "Training...") :
    t= Tournament(g,bagTrain)
    t.run()

t = Tournament(g, bag)
t.run()
print(t.matrix)



"""

for _ in tqdm(range(10000), desc = "Training...") :
    m = Meeting(g, QLearning(True), Tft(), length=2)
    m.run()

m = Meeting(g, QLearning(False), Tft(), length=2)
m.run()
m.prettyPrint()

QLearning.exportQTable()
"""



"""
t= Tournament(g,bag)        # default: length=1000
e= Ecological(t)            # default: pop=100
e.run()
e.tournament.matrix
e.historic
e.drawPlot()
"""