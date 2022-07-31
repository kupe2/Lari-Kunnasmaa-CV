import pandas as pd
import numpy as np
from pyautogui import *
import pyautogui
import time
import keyboard
import random
import win32api, win32con, win32gui
import glob
import sys, os
import re
import pygetwindow as gw
from tkinter import *
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.signal

random.seed(10)
# Can be removed

#690,30 1890
#690, 1050, 1890

# Screen parameters
screen_resoltion = [1920, 1080]
new_resolution = [960, 540]

# amount of random labs placed
repeat = 40000

#Amount of random maps created
random_maps = 50000



lab_imsize = [55,40]
tile_size = [60,60]
map_coords = [[690,30],[1890,30],[690,1050],[1890,1050]]

# The size of the map in 2D array
map_size = [20,17]



#Substract two lists
def sub_lists(list1, list2):
    array1 = np.array(list1)
    array2 = np.array(list2)
    subtracted_array = np.subtract(array1, array2)
    subtracted = list(subtracted_array)
    return subtracted

#  apply new resolution, not used, but can be applied easily, scalebility is important
def apply_new_resolution(new_resolution,lab_size, map_size):
        ratio = [new_resolution[0]/1920,new_resolution[1]/1080]
        lab_size =  [lab_size[x]*ratio[x] for x in range(2)]
        map_size =  [[i[x]*ratio[x] for x in range(2)] for i in map_size]
        return lab_size, map_size

#if screen_resoltion!= new_resolution:
#    lab_size, map_coords = apply_new_resolution(new_resolution, lab_size, map_coords)
#print(lab_size, map_coords)

# Stores the tile types, last one used to print heatmap
types= ["empty", "production", "upgrade", "dup"]
powers = [0,1,0,-1]

# Class for all of the tiles, the tiles only have two properties, power and type
class Unit():
    def __init__(self, unit_type):
        self.type = types[unit_type]
        self.power= float(powers[unit_type])

#Searches the map differnet types of tiles. And yes I do not you can do it a lot faster with npwhere or vectorize,
# But I want  use different ways of doing it, since this is only training
def Get_types(Map):
    production_coord = []
    upgrade_coord = []
    for i in range(Map.shape[0]):
        for j in range(Map.shape[1]):

            if (Map[i][j].type == "production"):
                production_coord.append([i, j])

            elif (Map[i][j].type == "upgrade"):
                upgrade_coord.append([i, j])

    pro_true = production_coord
    up_true = upgrade_coord

    return pro_true, up_true


vectorized_type = np.vectorize(lambda obj: obj.type) # Shows the types in the map
vectorized_power = np.vectorize(lambda obj: obj.power) # Shows the power a of the map



#Counts the amount of power each production cell has, this is based on te surronding upgrade tiles.
# There are several ways of doing this, this is probably the worst, but its fine it works.
# If i wanted to make something much better it would be this, scipy has a function for example
def count_power(production_coord, upgrade_coord,map ):
    for num_elem,elem in enumerate(production_coord):
        old_k = map[elem[0]][elem[1]].power
        k = 1

        for j in upgrade_coord:
            if ((elem[0] - j[0]) ** 2 + (elem[1] - j[1]) ** 2 <= 2 and map[elem[0]][elem[1]].power!=0 ):
                k = k+0.4 # This is the amount of power each addjacent upgrade cell give

        if float(k) != float(old_k): # Changes the power, with the addjacent upgrade cells
            map[elem[0]][elem[1]].power = float(k)

    return map


#Tries to search labs on the screen. This way we dont have to manully add them to a 2D array.
def get_labs(map):
    #Since the lab picture is so good a high confidence can be used
    labs = pyautogui.locateAllOnScreen("lab.png",confidence=0.99)
    off_set = [0,40] #For better calibration

    # This loop changes the coordinates, to positions in the 2D array
    for ind, lab in enumerate(labs):
        map[int((lab.left-map_coords[0][0]-off_set[0])/tile_size[0])][int((lab.top-off_set[1])/tile_size[0])] = Unit(1)

    return map.T



# This function may look ugly and that is because it is. But checking this is suprisnglt hard
# Scipy might have a very good function for this, but since this was suppose to be done via machine learning
# this is just a bad function that I made so I can show off my problem-solving when it comes to pyautogui
# Check for the most efficient places to place upgrade tiles on  the map.
def map_eff(map, rep):
    #

    all_tiles = np.argwhere(vectorized_type(map) != "empty")

    # To people who have weak hearts you might want to look a way this is too jank.
    for i in range(rep):
        coordinates = all_tiles[random.randint(0,len(all_tiles)-1)]
        r_coordinates = coordinates
        tile = map[coordinates[0],coordinates[1]]


        # The reason why this is so complicated is that the edges need to be taken in account.
        x = 0
        y=0
        if coordinates[0]== map_size[0]-1:
            coordinates[0]=18
            x=1
        if coordinates[1]== map_size[1]-1:
            coordinates[1]=16
            y=1
        if coordinates[0] == 0:
            coordinates[0]=1
            x=-1
        if coordinates[1]== 0:
            coordinates[1]=1
            y=-1

        # These if statements check if the tile should be a production tile or an upgrade tile. Why yes it does look like a monkeys
        # anus, but they work, since machine learning was supposed yo be used
        if tile.type== "production":
            if np.sum(vectorized_power(map[coordinates[0]-1:coordinates[0]+2,coordinates[1]-1:coordinates[1]+2])) \
                    < ((np.sum(vectorized_power(map[coordinates[0]-1:coordinates[0]+2,coordinates[1]-1:coordinates[1]+2])))-1-0.4*float(len(np.argwhere(vectorized_type(map[coordinates[0]-1:coordinates[0]+2,coordinates[1]-1:coordinates[1]+2]) == "upgrade"))))*1.4:
                map[r_coordinates[0]+x,r_coordinates[1]+y] = Unit(2)

        if tile.type== "upgrade":
            if np.sum(vectorized_power(map[coordinates[0]-1:coordinates[0]+2,coordinates[1]-1:coordinates[1]+2]))+1+0.4*float(len(np.argwhere(vectorized_type(map[coordinates[0]-1:coordinates[0]+2,coordinates[1]-1:coordinates[1]+2]) == "upgrade"))) \
                    > ((np.sum(vectorized_power(map[coordinates[0]-1:coordinates[0]+2,coordinates[1]-1:coordinates[1]+2]))))*1.4:
                map[r_coordinates[0]+x,r_coordinates[1]+y] = Unit(1)

    return map



# This funciton is used to create the histogram to compare the totally random place with my function.
def random_lab(filled_map, get_random):
    all_powers= np.array([])
    new_map = filled_map

    # This for loop creates new maps with a random variation on the upgrade tile posiotions.
    for i in range(get_random):
        coords = np.argwhere(vectorized_type(filled_map) == "production")


        # I found that 33% gives the map the highest production value
        choices = np.random.choice(coords.shape[0], size=int(len(coords)*0.33))

        # Fills the positions. Yes  I know it is a for loop and yes it does make me vomit as well
        for i in coords[choices]:
            filled_map[i[0],i[1]] = Unit(2)


        # Calculates the new power of the map and adds it to an array
        pro, up = Get_types(filled_map)
        filled_map = count_power(pro,up , filled_map)
        curre = np.sum(vectorized_power(new_map))
        all_powers = np.append(all_powers, curre)

        # Resets the map, yes for loop again disgusting
        for i in coords[choices]:
            filled_map[i[0],i[1]] = Unit(1)

    # Creates the histogram and plot for more appealing visuals
    _, bins, _  = plt.hist(all_powers, bins = 50, density= 1)
    plt.title("histogram of  maps power when beacons are placed randomly")
    mu, sigma = scipy.stats.norm.fit(all_powers)
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, best_fit_line)
    plt.show()

    return

# This function uses pyautogui to apply the insert the upgrade tiles in to the games map in the right positons.
def apply_map( upgrade_coord):
    Beacons = [330,110] # equips the beacons
    Beacon = [300,340]

    while keyboard.is_pressed('c') == False and win32gui.GetWindowText(
            win32gui.GetForegroundWindow()) == "NGU INDUSTRIES": # Always safety checks, you don't want you mouse to just randomly click

        # Selects the upgrade beacon from the other options
        pyautogui.moveTo(Beacons)
        pyautogui.click()
        pyautogui.moveTo(Beacon)
        pyautogui.click()

        # Places all of the upgrade tiles on their right places based on the screen coordinates
        for ind, cord in upgrade_coord:
            pyautogui.moveTo(60*cord+700,60*ind+40)
            pyautogui.click()

        break


if __name__ == '__main__':
    filled_map = np.full((map_size[0], map_size[1]),Unit(0))
    time.sleep(2)
    filled_map = get_labs(filled_map)

    # Shows what power would be expected if the upgrade tiles were placed at random.
    #if type(random_maps)==int:
    #    random_lab(filled_map, random_maps)

    # Calculates the most effective map placement
    filled_map = map_eff(filled_map, repeat)

    # Calculates the maps power and prints the total power of the map
    pro, up = Get_types(filled_map)
    filled_map = count_power(pro, up, filled_map)
    print(np.sum(vectorized_power(filled_map)))

    # applyies the most effective map on to the game map using pyautogui
    apply_map(up)

    # To visualize it better we create a heatmap, with the upgrade tiles as -1 values.
    filled_map = np.where(vectorized_type(filled_map) == "upgrade", Unit(3), filled_map)
    sns.heatmap(vectorized_power(filled_map), annot=True)
    plt.show()

