'''
web scrapping sample code
'''

from splinter import Browser
from splinter.exceptions import ElementDoesNotExist
import numpy as np
import pandas as pd
import time

import secrets

# ================== input values ==================

selected_play = 'Bakken Shale'
drilling_days = 30

# ================== functions ==================

def init_browser():
    executable_path = {'executable_path': 'chromedriver.exe'}
    return Browser('chrome', **executable_path, headless=False)

def login(browser: Browser) -> None:
    '''
    System login
    '''
    login_url = 'http://spowertrain-orange.ihsmarkit.com/Account/Login'
    costanalyst_url = 'http://spowertrain-orange.ihsmarkit.com/Account/RedirectTo?returnUrl=http%3A%2F%2Fsna-powertrain-orange.ihsmarkit.com%2F'

    # login
    browser.visit(login_url)
    browser.find_by_id('UserName').fill(secrets.login_creds['username'])
    browser.find_by_id('Password').fill(secrets.login_creds['password'])
    browser.execute_script("document.getElementsByTagName('form')[1].submit()")
    time.sleep(2) # wait for content to load

    # visit cost analysis page
    browser.visit(costanalyst_url)
    time.sleep(2) # wait for content to load
    # click on 'Cost Analysis Grid' button
    tries = 0
    while True:
        tries += 1
        if tries > 5:
            raise Exception('Failed to click on "Cost Analysis Grid" button')
        try:
            browser.execute_script('[].slice.call(document.querySelectorAll("button")).filter(b => b.textContent.match(" Cost Analysis Grid"))[0].click()')
            break
        except Exception as e:
            # sleep and try again later
            # TODO: log the error
            print(e)
            time.sleep(2)
    time.sleep(2) # wait for content to load

def set_play(browser: Browser, play: str) -> None:
    '''
    Set Play/Subplay and navigate to Well Data panel
    '''
    # select Play/Subplay
    play_dropdown = browser.find_by_css('div[class="cost-analysis-grid-header-play-subplay"]').first.find_by_tag('select').first
    play_dropdown.find_by_text(play).click()
    time.sleep(2) # wait for content to load
    # click on Well Data pencil button
    browser.find_by_css('i[class="fa fa-pencil-square fa-2x"]').first.click()
    time.sleep(2) # wait for content to load

def set_drilling_days(browser: Browser, drilling_days: int) -> dict:
    '''
    Set drilling days and return results
    '''
    # go to Well Data tab
    browser.find_by_text('Well Data').click()
    # set drilling days and hit Save
    browser.find_by_id('DrillingDays').fill(drilling_days)
    browser.find_by_text('Save').click()
    time.sleep(2) # wait for content to save
    tries = 0
    while True:
        tries += 1
        if tries > 5:
            raise Exception('Failed to dismiss Alert message')
        try:
            alert = browser.get_alert()
            alert.accept()
            break
        except Exception as e:
            # sleep and try again later
            # TODO: log the error
            print(e)
            time.sleep(2)

    # click on Cost Summary and return results
    browser.find_by_text('Cost Summary').click()
    total_capex = browser.find_by_id('curCapexTotalThis').value
    drilling_total = browser.find_by_id('curDrillTotalThis').value

    return {'Total Capex' : total_capex, 'Drilling Total': drilling_total}

# ================== main code ==================

if __name__ == "__main__":
    browser = init_browser()
    login(browser)
    set_play(browser, selected_play)
    
    for drilling_day in [20, 35, 40]:
        result = set_drilling_days(browser, drilling_day)
        print(f'Drilling Days: {drilling_day} - {result}')