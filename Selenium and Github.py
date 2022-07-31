from github import Github
import os
import Vault
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()

# Type the molecule of which mass you want to know

name = "kupe2"
project = "Lari-Kunnasmaa-CV"
company = "Acme" #
git_traffic = False
save = True
close = False


# I use this function in my thesis, to find molecules.
# This speeds up the process of googling the molecules accurate mass and it also opens its mass spectrum if it has one.
# This way I dont have to manually look for it saving a lot of time when dealing with a lot of different molecules
def get_mol_mass(molecule):
    driver = webdriver.Chrome()
    driver.get(f"https://pubchem.ncbi.nlm.nih.gov/compound/{molecule}")  # Goes  to the page

    # Tries to find the mass of the molecule, if it does not find it will search the molecule, by click the search.
    # This function should be rewritten so that it would still inform me that the page exist, but there is no mass, but since I 100 % need the mass this is not necessary.

    try:
        Mol_mass = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="Computed-Properties"]/div[2]/div/div[1]/table/tbody/tr[1]/td[2]'))) # The path  of the mass is always the  same so this works fine

        # Tries to look if the are any data on the mass spectrometry, if there is open the "page"
        try:
            Mol_peak = WebDriverWait(driver, 0.5).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="GC-MS"]/div[1]/div[1]/div/a/span'))) # The MS Data is alway in the same place..
            Mol_peak.click() # opens MS data
            print(f"{molecule} has a mass of {Mol_mass.text} and there is MS data, open page") # Prints this jsut for fun
            return Mol_mass.text

        except:
            print(f"{molecule} has a mass of, but no MS data")
            return Mol_mass.text

    except:
        # Because, the molecule might still exist  I want to see what possible pages there might exist for it.
        # for example  2,5-dibromo-4-nitro-H-imidazole is the common name used for the molecule, but in pubchem  it is called 2,5-dibromo-4-nitro-1H-imidazole
        # this makes it easy to click the molecule you want.
        print(f" You have misspelled the molecule {molecule}, opening search page to find it") # Print to let me know waht happend

        new_se = driver.find_element(By.XPATH, '//*[@id="main-content"]/div/div/div[3]/a') # Clicks the search bar
        new_se.click()


# Gets the amount of unique views from my Github CV. This is to track what companies actually look at my github, why yes I do keep track of the data.
# No worries it will never be published and if you actually reading this means you are one of the good companies. So, good for you.
def get_traffic(name, project):

    driver.get(f"https://github.com/{name}/{project}") # Looks for the specific project

    # Token is stored in an encrypted  vault. Did you actually think  that I would just store it here and upload it to github?
    # I might be an idiot, but there is one thing I am not and that is an idiot :D.
    token = os.getenv('GITHUB_TOKEN', Vault.GitHub_token)
    ghub = Github(token)

    # Checks the amount of times the page has been viewed.
    repo = ghub.get_repo(f"{name}/{project}")
    days_views = repo.get_views_traffic(per="day")

    return days_views

 # Tracks the companies, that have looked at the github. So I can report it to Mr "joulupukki" if the company has been naughty or nice.
 # note (if you looked at github you are nice)
 # I should really add an automatic email sent to santas' workshops
def save_traffic_to_excel(days_views):
    Git_jobs = pd.read_excel('Git_jobs.xlsx', index_col=0)
    Git_jobs.loc[company, "Viewed"] = True
    Git_jobs.loc[company, "Git_looked"] = days_views['views'][-1].timestamp
    Git_jobs.to_excel("Git_jobs.xlsx")
    return


# So, I felt very inspired after combining my github and molecule codes so, I wanted to test my molecule search code.
# So I wrote this little beast that goes to a website to get a random molecule and then searches it in pubchem after a 100 runs did not crash. I would call that "iha jees"
#
def get_molecule():
    driver = webdriver.Chrome()
    driver.get("https://www.zuiveringstechnieken.nl/en-gb/random-hydrocarbons")
    mol  =driver.find_element(By.XPATH, '//*[@id="1920353114"]/font/center/b')
    mol_name = mol.text.split(",")[1]
    driver.close()
    return mol_name

molecule = ""
repeats = 100

if __name__ == '__main__':
    # This checks the molecules mass and massspectrum I use this in my thesis to optimize workflow
    if len(molecule) > 0:
        mol_mass = get_mol_mass(molecule)

    # This is a loop creating a random molecule and checking it, because I why not.
    for i in range(repeats):
        mol_name = get_molecule()
        time.sleep(2)
        mol_mass =get_mol_mass(mol_name)
        print(mol_mass)
        driver.quit()

    # Checks git traffic
    if git_traffic:
        days_views = get_traffic(name, project)
        print(days_views["uniques"])
        if save and days_views['uniques'] != days_views['views'][-1].uniques:
            save_traffic_to_excel(days_views)
    get_molecule()

    if close:
        driver.quit()



