"""my_supervisor controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor, Receiver, Keyboard, Node

# create the Robot instance.
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
nao = supervisor.getFromDef("NAO")
reciever = supervisor.getDevice("receiver")
reciever.enable(timestep)
reciever.setChannel(1)


while supervisor.step(timestep) != -1: 
    if reciever.getQueueLength()>0 :
        #print(reciever.getData().decode('utf-8'))
        nao.loadState("__init__")
    reciever.nextPacket()
    