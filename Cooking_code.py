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

"""
Hello Welcome to the project 


"""

path = "D:/Python/Side_project/Ngu/numbers"

# Creates a list to save images
image_paths = []
image_paths.append(os.path.join(path, ".png"))

to9 = list(range(0, 10))
to9 = [str(x) + ".png" for x in to9]


# This  function reads all of the pictures taken from the sample numbers
def read_image(path, directories: object, images):
    for dir in directories:
        folder_path = os.path.join(path, dir)
        for im in images:
            if os.path.isfile(
                    os.path.join(folder_path, im)):  # Try except could also be an option, but this helps with debugging
                image_paths.append(os.path.join(folder_path, im))
            else:
                print(im, " not in " + dir)
    print(image_paths)
    return image_paths


# This function is used check the CME percentage after the amount of ingredients have changed.
# AS stated in the pdf we need image recognition to get the number, Python OCR package  was little unreliable in this case so pyautogui was used.
def fast_cooking():
    res = 0
    if win32gui.GetWindowText(win32gui.GetForegroundWindow()) == "NGU Idle":  # Again check if we are in correct window
        CME = pyautogui.locateOnScreen("CME.png",
                                       confidence=0.9)  # Search for the CME position again to  check if the window has been moved.
        numb = pd.DataFrame(columns=["num"])  # Place to store the CME value

        for num, x in enumerate(to9):  # Goes throught the images 0 to 9:
            # Because we checked the CME position earlier we can check for the numbers at a specific  position.
            Num_found = locateAllOnScreen(os.path.join(path, x), region=(CME.left + CME.width, CME.top - 5, 150, 35),
                                          confidence=0.9)  # confidence 0.9 has been enough and does not cause any issues

            if Num_found != None:  # if we found a number then its position is saved.
                for i in Num_found:
                    numb.loc[i.left] = num

            else:
                continue

        # This function arranges the number is the correct order. So if the CME value is 4204 the dataframe looks like 0244
        #  But because the index tracks their coordinates, when sorting by index we get the 4204
        # Is this the best solution? Oh god NO! Is it a solution that works 100 % of the time? Yes

        numb = numb.sort_index()
        s = [str(integer) for integer in numb["num"].values]  # Just combining the numbers and converting the str to int
        a_string = "".join(s)
        res = int(a_string)

    else:  # fail safe
        print("You have clicked out, therefore stopping  fast cooking")
        sys.exit()

    return res


# The created ingredient class with their x,y position on the screen. The positions are base on CME postion so even if the window is in different place the code works!
class ingredient:
    def __init__(self, nu, loca):
        self.x = loca.left + 75 + (500 * math.ceil(nu % 2))
        self.y = loca.top - 626 + (156 * math.floor(nu / 2))

        self.current = 0  # current ingredient values
        self.max = 0  # Ingredient number where the CME is at its max


# Just a function to reset all ingredient values to 0, to start the search
def reset_all(ingredients):
    for inde, x in enumerate(ingredients):  # Goes throught all the ingredients we want to check
        if win32gui.GetWindowText(
                win32gui.GetForegroundWindow()) == "NGU Idle":  # Safety is key. You dont want the program to click 140 times in random places. (köh köh)
            click(ingredients[inde].x, ingredients[inde].y + 40, clicks=20,
                  interval=0.05)  # Clicks 20 on the minus button of the ingredient
        else:  #
            print("Reset could not be finished, because window does not exist")
            sys.exit()
    return


def click_to_max(ingredient, max_number):  # When the maximum of the ingredient has been found this function clicks to that position
    if win32gui.GetWindowText(win32gui.GetForegroundWindow()) == "NGU Idle":  # Safety check
        click(ingredient.x, ingredient.y + 40, clicks=max_number - ingredient.max,
              interval=0.05)  # Clicks to the maximum value
    else:
        print("Click not done since window not active")
        sys.exit()
    return


# Just some miportant coordinates
# point = x=1069, y=216  1559  372
# x = 372-216 = 156
# 992 840 CME


Repeats = 1
amount_of_ingredients = 7
max_number = 20
Reset = True

# time.sleep(1)
# print(pyautogui.position())

if __name__ == '__main__':
    time.sleep(1)
    while keyboard.is_pressed('c') == False and win32gui.GetWindowText(
            win32gui.GetForegroundWindow()) == "NGU Idle":  # Emergency shutoff. Has NEVER been used since I am so good at coding and lying :D

        # This loop is useless since it only does it once, but I believe that code should be as scalable as possible. So it is there just incase it needs to repeated.
        for i in range(Repeats):

            CME = pyautogui.locateOnScreen("CME.png",confidence=0.9)  # Because the window can be in any part of the screen we this to locate it.

            ingredients = {x: ingredient(x, CME) for x in range(amount_of_ingredients)}  # We create a dictinary to store the ingredient class
            #
            if Reset:
                reset_all(ingredients)

            for ind, num in enumerate(ingredients): #
                max_res = 0

                for switch in range(max_number):
                    current = fast_cooking()
                    if current > max_res:
                        max_res = current
                        ingredients[ind].max = switch
                    click(ingredients[ind].x, ingredients[ind].y)
                click_to_max(ingredients[ind], max_number)

