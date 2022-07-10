import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib .use('TkAgg')
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
from random import gauss
from random import randrange
import time


start_time = time.time() # just to check the run time

class Particle:

    def __init__(self,   type,position, velocity  ,diffusion_constant):
        self.type = type
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.diffusion_constant = diffusion_constant

    def Update_position(self, Time):
        self.position = self.position + self.velocity * Time




maps=[] #Saves the generated heatmaps
i =0
max_values= []

Amount_steps = 100 # amount of seconds to run the simulation should be divisable with between steps variable
Between_steps =  20#Seconds between simulations
steps = int(Amount_steps / Between_steps) # The amount of steps

Pipex=20# Pipes witdh
Pipey=20 #Pipes lenght
per = 3

# How many particles fit on one x and y unit should be around 2 or 3 in order to avoid white sp
open = True # The simulation type pipe/box


# The Position, Velocity, Force are all defined as [x,y]
Velocity= [0,0.22] #Flow rate of the pipe


D_solv = 10**-3      # The diffusion constant of the solvent
Temperatures= [300, 500] # Temperatures of the fluids solvent first temp in kelvins and solute last
Viscosity = [7.972*10**-10, 0.4] #Viscosities of the fluids solvent first and solute last
R_0 =8.4 *10**-6 #Radius of the dye particle
k_B= 1.38064*10**-17

# The function creates the a matrix which is
def heatmat(particles, maps):
    particle_DF = pd.DataFrame()
    Heatmap = np.zeros((Pipex,Pipey)) #Heatmap prep

    for part in particles:
        particle_DF= particle_DF.append([[part.position[0],part.position[1],part.type]])
    particle_DF.columns = ["x","y","t"] # Particles are placed in the dataframe with these columns for easier data management

    # Goes throught all of the x,y coordinates and calculates the amount of particles in each square
    # and  calculates their types mean, which determines the color of the square.
    for x_coord in np.arange(Pipex):
        for y_coord in np.arange(Pipey):
            safe = particle_DF[(y_coord+1>particle_DF.x ) & (y_coord<particle_DF.x)&(x_coord+1>particle_DF.y ) & (x_coord<particle_DF.y)  ]
            Heatmap[x_coord][y_coord] = safe.t.mean()


    where_are_NaNs = np.isnan(Heatmap)
    Heatmap[where_are_NaNs] = 1# This replaces the squares which has no particles with the solvent.
    # This step could be avoided with a high probability by setting the per variable over 3, but this increases the the runtime, with little effect in the simulation.

    maps.append(Heatmap)# adds the heatmaps for the slideshow
    max_values.append(np.max(Heatmap)) #Max values to thelp with the slideshow
    return

#Mostly used for debbugging
def show_particles(particles):

    xwall =  [0,0,Pipex,Pipex,0]
    ywall = [0,Pipey,Pipey,0,0]# The box
    xfluid= []
    yfluid =[]
    xcolor = []
    ycolor = [] #Prep

    for part in particles:
        if part.type==2:
            xcolor.append(part.position[0])
            ycolor.append(part.position[1])
        elif part.type==1:
            xfluid.append(part.position[0])
            yfluid.append(part.position[1])
        else:
            xwall.append(part.position[0])
            ywall.append(part.position[1]) #Plots each type on its own

    plt.plot(xcolor,ycolor, 'o', label="color" )
    plt.plot(xfluid, yfluid, 'o', label="fluid")
    plt.plot(xwall, ywall, label="wall")
    plt.legend()
    plt.show()





#Moves the particles based on their velocity
def Move_particles(particles, dt):

    for part in particles:
        part.Update_position(dt)


def add_walls(particles):

    # loop over all particles
    for i in range(len(particles)):

        if particles[i].position[0] < 0:
            # If particle is over the right wall it bounces from it in an elastic collision and its velocity also changes to match this event
            particles[i].position[0] = -particles[i].position[0]
            particles[i].velocity[0] = -particles[i].velocity[0]
            pass

        if particles[i].position[0] >Pipex:
            # If particle is over the left wall it bounces from it in an elastic collision and its velocity also changes to match this event
            particles[i].position[0] = (2*Pipex-particles[i].position[0])
            particles[i].velocity[0] = -particles[i].velocity[0]
            pass

#The function checks the pipe type and sets the correct constraints.
def Pipe_type(particles, open):

    for i in range(len(particles)):

        if(open): # Check the system type
            if particles[i].position[1] >= Pipey:
                particles[i].position[1] = particles[i].position[1]-Pipey
                # If particle is over the pipe it is generated in the beginning to keep the flow rate constant

                if particles[i].type == 3: # Checks the type and if it is dyed then it is changed to a regular particle inorder to keep the flow rate constant
                    particles[i].type = 1

            if particles[i].position[1] <= 0:
                particles[i].position[1] = Pipey+particles[i].position[1]

                # If particle is over the pipe it is generated in the beginning to keep the flow rate constant
                if particles[i].type == 3:# Checks the type and if it is dyed then it is changed to a regular particle inorder to keep the flow rate constant
                    particles[i].type = 1

        else:
            if particles[i].position[1] > Pipey: # If The pipe is a box then the ends are treated the same way as the wall.
                particles[i].position[1] = 2*Pipey - particles[i].position[1]
                particles[i].velocity[1] = -particles[i].velocity[1]


            if particles[i].position[1] < 0:
                particles[i].position[1] = -particles[i].position[1]
                particles[i].velocity[1] = -particles[i].velocity[1]


def Particle_movement(particles, dt, time,  open, maps):

    Pipe_type(particles, open) # Checks pipe type (pipe/box)
    add_walls(particles)       # Adds the walls so that the simulation does not go overboard


    particles = Brownian_dis(particles, time)
    # Calculates the brownian distance and assign a velocity based on this distance every particle

    for i in range(steps):
        if (i==0):
            heatmat(particles, maps)
        Move_particles(particles, dt) #Updates the position of the particles based on their velocity

        Pipe_type(particles, open)
        add_walls(particles)    # These functions enforce the systems limits like walls.
        heatmat(particles, maps)





    return particles


# We can check the momentum of the systems using this function
def calculate_momentum(particles):
    p = np.array([0.0, 0.0])
    for part in particles:
        p += part.mass * part.velocity
    return p


    #The dyes diffusion constant is defined. It is done using the formula D_dye = T_1/ T_0 * viscosity_0 / viscosity_1* diffusion constant of the solvent
def Dif_const(Temperatures, D_solv, Viscosity):
    D_dye=Temperatures[1]/Temperatures[0]*(Viscosity[0])/Viscosity[1]* D_solv
    #print(D_dye)
    return D_dye


def Dif_con(Temperatures, Viscosity):
    D_dye=(k_B*Temperatures[1])/(6*3.14*Viscosity[0]*R_0)
    return D_dye

# Calculates the brownian distance and assings a velocity to the particles based on this velocity
def Brownian_dis(particles, time):

    for i in range(len(particles)):
        particles[i].velocity[0] = particles[i].velocity[0]+(2*particles[i].diffusion_constant*time)**(1/2)*gauss(0, 1.27)/time # Adds the brownian velocity to the x velocity.
        # Gauss distribution is used to give each particle unique velocity and other random determines the direction of the movement in the axis.
        #To achieve bell curve random values were tried until the absolute value mean distance was 1.

        particles[i].velocity[1] = particles[i].velocity[1]+(particles[i].diffusion_constant*2*time)**(1/2)*gauss(0, 1.27)/time

        print(particles[i].velocity[0])
        #Same thing for the y axis

        # The velocity is calculated using brownian distance and dividing it with the simulation time for each coordinate separatly. d|x|/dt =  dsqrt(2*D*t)/dt  Is the formula used for the x velocity.
        #We use velocity so we can see the evolution of the system.
    return particles



def Pipe(per, width, lenght, Velocity, D_solv, Viscosity):
    width =(width)*per
    lenght=(lenght)*per
    Pipe= np.zeros((width+1, lenght+1)) #Creates the pipe matrix based on the dimensions given

    Pipe[:width,:lenght ]= 1 # Fill the Pipe matrix fit "fluid"
    Pipe[:width,3:6] = 2 # The dye/other liquid is inserted


    amount = []
    for j in range(1,3): #
        whe = np.where(Pipe==j) #Searhes the pipe for both fluids

        for i in range(len(whe[0])): #Goes through the list of found particles.

            position = np.array([0.0] * 2) #Positions and velocities first coordinate is the x  coordinate.
            velocity = np.array([0.0] * 2)
            position[0] = whe[0][i]/per
            position[1] = whe[1][i]/per
            velocity[0] = Velocity[0]
            velocity[1] = Velocity[1]  #Gives the particles their velocity and position

            type = j# Gives the particle its type and mass


            if(j!= 1): # Diffusion constant for particle depending on its type

                #diffusion_constant = Dif_const(Temperatures,D_solv,Viscosity)
                diffusion_constant=Dif_con(Temperatures, Viscosity )

                print(diffusion_constant)
            else:
                diffusion_constant = D_solv

            amount.append(Particle(type,position, velocity,diffusion_constant ))
            #Adds the partic

    return amount



amount = Pipe(per, Pipex, Pipex, Velocity, D_solv, Viscosity)
# The function constructs the pipe and defines the particles


particles=Particle_movement(amount, Between_steps, Amount_steps, open,maps)
#The function performs the brownian motion on the defined system.


Slideshow = plt.figure()
Slideshow.add_subplot(111) #Slide show prep


def Generate_map(fig):

    cdict = {'red': ((0.0, 0.0, 0.0),  # Gives is the amount of red color when there are no dye particles
                     (0.5, 0.0, 0.0),  # Gives is the amount of red color when there are the same amount of particles
                     (1.0, 0.0, 0.0)),  # Gives is the amount of red color when there are only dye particles

             'green': ((0.0, 0, 0),  # same as red but for green
                       (0.5, 1, 1),
                       (1.0, 0.5*max_values[i], 0.5*max_values[i])), # Decarease as the maximum value decrease. Not the best way of doing it, but gets the point across well

             'blue': ((0, 1, 1),  # same as red but for blue
                      (0.5, 1, 1),
                      (1,2-max_values[i] , 2-max_values[i] ))
             }  # Defining the colors used here we can see that the solvent used blue and the dye is green. Therefor their mixed color is somewhere in between.
    # The choice of the color was done pretty randomly and  I

    heat_map = sns.heatmap(maps[i], cmap= mcolors.LinearSegmentedColormap('GnBu', cdict))# PLots the heatmap using the defined colors
    heat_map.invert_yaxis() # Axis needs to be corrected


def Generated_maps():
    maps = dict()   # We create a dictanary for all of the generated heat maps and return it in order to use them on the show
    for map in range(steps):
        maps[map]=Generate_map
        print(maps)
    return maps


#On click the function shows the next map from the dictanary
def On_click(fig):
    global i
    i += 1
    i %= steps
    fig.clear()
    print("Figure",i)
    Generated_maps()[i](fig)
    plt.draw()

print(Generated_maps()[0])
Generated_maps()[0](Slideshow) #Starts the show with 0 element of dictinary
Slideshow.canvas.mpl_connect('button_press_event', lambda event: On_click(Slideshow))
#Changes the map when clicked

plt.show()#Shows the plot
#show_particles(particles)

print("--- %s seconds ---" % (time.time() - start_time))
