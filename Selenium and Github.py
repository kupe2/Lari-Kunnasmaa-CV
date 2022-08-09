from github import Github
import os
import Vault
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import imaplib
import email
from email.header import decode_header
import webbrowser









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
# No worries it will never be published and if you actually reading this means you are one of the good companies. So, good for you guys :D.
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
    driver.get("https://www.zuiveringstechnieken.nl/en-gb/random-hydrocarbons") # Goes to a site and gets a random hydrocrabon
    mol  =driver.find_element(By.XPATH, '//*[@id="1920353114"]/font/center/b')
    mol_name = mol.text.split(",")[1]
    driver.close()
    return mol_name



def read_email():
    # Large credit goes to https://www.thepythoncode.com/article/reading-emails-in-python, who has provided excellentt documentation on the subject
    imap = imaplib.IMAP4_SSL(server, 993) # IMAP class
    print(f"Connection Object : {imap}") #Check connection

    imap.login(Vault.email_name, Vault.password)  # Again password in vault, so that you cannot see it :D
    imap.select() # Select inbox
    status ,messages = imap.search(None, "UNSEEN") #Search for  so unseen emails

    for message in messages[0].split(): # Check all the messages
        typ, datas = imap.fetch(message,'(RFC822)')  # Get emails with ID,


        for data in datas: #Gets the
            if isinstance(data, tuple):
                msg = email.message_from_bytes(data[1])
                From = msg["From"] # Gets who sent the message
                subject = msg["subject"]  # Gets subject
                print(f"A message from {From} about {subject}")

                for part in msg.walk(): #Prints the message
                        if part.get_content_type()=="text/plain" :
                            messages = part.get_payload(decode=True)
                            print(messages.decode())


    imap.close()
    imap.logout()
    return From.replace("<", '').replace(">",'').split()[-1]   
 
#Just a simple email reply if the email is sent, feel free to test it out.
def send_reply(From):
    #Just a simple text reply, wanted to put some funny images for example, but since if someone actually does want to 
    # test the code did not want to make it too much of a security risk. Althought I might add a text cat un the future
    # Or use sklearn to try to interpert the original message sent.
    e_mail = Vault.email_name
    SUBJECT = "An automatic reply from from the CV email code"
    TEXT = """\
    Hello,\n
    This has been an automatic reply :D\n 
    I thank you for your participation 

    """
    email_text = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT) 
    sgma = smtplib.SMTP('smtp.gmail.com', 587)# Opens gmail
    sgma.ehlo() # 

    sgma.starttls() # The sent to the address that sent the original message 
    sgma.login(Vault.email_name,Vault.password)
    sgma.sendmail(e_mail, From, email_text)
    print("Send to %s success!" % From)
    sgma.quit()

    return




name = "kupe2"
project = "Lari-Kunnasmaa-CV"
company = "Acme"
git_traffic = False
save = True
close = False
molecule = ""
repeats = 100

server = "imap.gmail.com"

reply= True

if __name__ == '__main__':
    # I made  small script that reads my fake accounts gmail,
    # Now when running the code it checks for new messages and replies to all new messsages
    read_email()
    if reply:
        send_reply(From)


    driver = webdriver.Chrome()
    #This checks the molecules mass and massspectrum I use this in my thesis to optimize workflow
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



