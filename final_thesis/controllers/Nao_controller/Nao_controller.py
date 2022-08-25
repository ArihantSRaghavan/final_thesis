"""Nao_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Emitter, Keyboard
from math import pi, sin, sqrt, exp
import math
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from math import pi, sin
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

emitter = robot.getDevice("emitter")
emitter.setChannel(1)


gps = robot.getDevice('gps')
gps.enable(timestep)

gyro = robot.getDevice('gyro')
gyro.enable(timestep)

l_gps = robot.getDevice('l_gps')
l_gps.enable(timestep)

r_gps = robot.getDevice('r_gps')
r_gps.enable(timestep)

RShoulderPitchS = robot.getDevice("RShoulderPitchS")
RShoulderPitchS.enable(timestep)

LShoulderPitch = robot.getDevice("LShoulderPitch")
RShoulderPitch = robot.getDevice("RShoulderPitch")

LShoulderPitch.setPosition(1.6)
RShoulderPitch.setPosition(1.6)
actuators=["RKneePitch","LKneePitch","RHipPitch","LHipPitch","RAnklePitch","LAnklePitch","RHipRoll","LHipRoll","RAnkleRoll","LAnkleRoll"]
actuator_m=[]

def sigmoid(x):
	return 1.0 / (1.0 + exp(-.3*x))

def reset_hand():
    while robot.step(timestep) != -1 :
        LShoulderPitch = robot.getDevice("LShoulderPitch")
        RShoulderPitch = robot.getDevice("RShoulderPitch")
    
        LShoulderPitch.setPosition(1.6)
        RShoulderPitch.setPosition(1.6)
        
        check = RShoulderPitchS.getValue()
        #print(check)
        if check <= 1.6 :
            #print("reset done")
            break
        
for i in actuators:
    actuator_m.append(robot.getDevice(i))


def chromo():
    return np.random.uniform(-0.1,0.2)
    
def position():
    p0 = l_gps.getValues()
    l_translation = [float(j) for j in p0]
    p1 = r_gps.getValues()
    r_translation = [float(j) for j in p1]
    translation = []
    x=(l_translation[0]+r_translation[0])/2
    y=(l_translation[0]+r_translation[0])/2
    translation.append(x)
    translation.append(y)
    return translation
  
def motion(ind):
    actuators=["LKneePitch","RKneePitch","RHipPitch","LHipPitch","RAnklePitch","LAnklePitch","RHipRoll","LHipRoll","RAnkleRoll","LAnkleRoll"]
    actuator_m=[]
    #print(ind)
    start = robot.getTime()
    
    #print("Start time=",start)
    for i in actuators:
        actuator_m.append(robot.getDevice(i))
    #print(timestep)
    ti=0
    F=1
    n=-15
    while robot.step(timestep) != -1 :
        j=0
        for i in range(0, len(ind),3):
            a=ind[i]
            b=ind[i+1]
            c=ind[i+2]
            if j < 2 :
                position = 1.3*c+2*sigmoid(n)*a*sin((ti-b) * 2.0 * pi * F)
            elif j >1 and j < 4:
                position = c+(-2)*sigmoid(n)*a*sin((ti-b) * 2.0 * pi * F)
            
            else :
                position = c+sigmoid(n)*a*sin((ti-b) * 2.0 * pi * F)
            actuator_m[j].setPosition(position)
            j+=1
        ti += timestep / 1000.0
        n +=1
        p = gps.getValues()
        z = float(p[2])
        lapse_time=robot.getTime()-start
        if z < 0.15 or lapse_time > 9:
            t=robot.getTime() - start
            position_1=l_gps.getValues()
            #print("Lapsed time =",t)
            return t

def reload_state():
    while robot.step(timestep) != -1 :
        emitter.send("Reload".encode('utf-8'))
        p = gps.getValues()
        z = float(p[2])
        if z > 0.1 :
            reset_hand()
            break
            



def evaluate(ind):
    init_pos = position()
    #print("Running Evaluator")
    #ind = [item for sublist in ind for item in sublist]
    ti = motion(ind)
    ti = float(ti)
    final_pos = position()
    p = gps.getValues()
    D=[float(j) for j in p] 
    tot_distX = final_pos[0] -init_pos[0]
    tot_distY = abs(final_pos[1] -init_pos[1])
    #print(tot_distY)
    tot_D = sqrt(tot_distX**2+tot_distY**2)
    tot_distZ = 0.334-D[2]
    f= 1+10*ti/9+20*tot_D+50*tot_distX-10*tot_distY
    #print("Done Calculation")
    reload_state()
    #print("Fitness",f)
    if math.isnan(f):
        return 0,
    else :
        return f,
            


 
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("chromo", chromo)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.chromo, n=30)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform,indpb=0.75)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selRoulette)


def main():
    reset_hand()
    #print("strarting main")
    POPULATION_SIZE = 60
    P_CROSSOVER = 0.8  # probability for crossover
    P_MUTATION = 0.02  # probability for mutating an individual
    MAX_GENERATIONS = 30
    HALL_OF_FAME_SIZE = 10
    runs=3
    maxList = []
    avgList = []
    minList = []
    stdList = []
    
    
    
    for r in range(0,runs):     
        pop=toolbox.population(n=POPULATION_SIZE)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    
        print("\n\nCurrently on run", r, "of",runs)
        #print("Starting Evolution")
        population, logbook = algorithms.eaSimple(pop, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,ngen=MAX_GENERATIONS,stats=stats, halloffame=hof, verbose=True)
        # print Hall of Fame info:
        df_log = pd.DataFrame(logbook)
        if r == 0 :
            df_log.to_csv('/home/arihant/Documents/logbook_1.csv', index=False)
        if r == 1 :
            df_log.to_csv('/home/arihant/Documents/logbook_2.csv', index=False)
        if r == 2:
            df_log.to_csv('/home/arihant/Documents/logbook_3.csv', index=False)
        #df_log.to_csv('/home/arihant/Documents/gendata2.csv', index=False)
        print("Hall of Fame Individuals = ", *hof.items, sep="\n")
        print("Best Ever Individual = ", hof.items[0])
        # Genetic Algorithm is done - extract statistics:
        meanFitnessValues, stdFitnessValues, minFitnessValues, maxFitnessValues  = logbook.select("avg", "std", "min", "max")
  
  
  
  
  
  
  
  
  
  
        # Save statistics for this run:
        avgList.append(meanFitnessValues)
        stdList.append(stdFitnessValues)
        minList.append(minFitnessValues)
        maxList.append (maxFitnessValues)
# Genetic Algorithm is done - plot statistics:
#sns.set_style("whitegrid")
    x = np.arange(0, MAX_GENERATIONS+1)
    avgArray = np.array(avgList)
    stdArray = np.array(stdList)
    minArray = np.array(minList)
    maxArray = np.array(maxList)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Max and Average Fitness for 3 Runs')
    plt.errorbar(x, avgArray.mean(0), yerr=stdArray.mean(0),label="Average",color="Red")
    plt.errorbar(x, maxArray.mean(0), yerr=maxArray.std(0),label="Best", color="Green")
    plt.show()
    

    

main()