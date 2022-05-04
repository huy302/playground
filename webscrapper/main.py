'''
web scrapping sample code
'''

from splinter import Browser
from selenium.common.exceptions import ElementClickInterceptedException
import time

import secrets

# ================== input variables ==================

selected_play = 'Bakken Shale'
drilling_days = [20, 35, 40]
mod_factors = [1.5, 1.75, 2]

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
    browser.find_by_text(' Cost Analysis Grid').click()
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

def set_drilling_days(browser: Browser, drilling_days: int) -> None:
    '''
    Set drilling days and save
    '''
    # go to Well Data tab
    browser.find_by_text('Well Data').click()
    # set drilling days and hit Save
    browser.find_by_id('DrillingDays').fill(drilling_days)
    browser.find_by_text('Save').click()
    time.sleep(2) # wait for content to save
    alert = browser.get_alert()
    alert.accept()

def set_mod_factor(browser: Browser, mod_factor: float) -> None:
    '''
    Set modification factor and save
    '''
    # go to Drilling Cost tab
    browser.find_by_text('Drilling Cost').first.click()
    # edit Permit and Survey cost sheet
    browser.find_by_text('Permit and Survey').first.find_by_xpath('..').first.find_by_css('i[class="fa fa-pencil-square-o fa-2x"]').first.click()
    # set modification factor, comments
    browser.find_by_id('ModificationFactor').fill(str(mod_factor))
    if mod_factor != 1:
        browser.find_by_id('modCommentsForEdit').fill('Automated service')
    # try to hit Save, if Save is disabled then hit Cancel
    try:
        browser.find_by_text('Save').click()
    except ElementClickInterceptedException:
        browser.find_by_text('Cancel').click()
    time.sleep(1) # wait for content to save

def extract_cost(browser: Browser) -> dict:
    '''
    Click on Cost Summary and return results
    '''
    browser.find_by_text('Cost Summary').click()
    total_capex = browser.find_by_id('curCapexTotalThis').value
    drilling_total = browser.find_by_id('curDrillTotalThis').value
    completion_total = browser.find_by_id('curCompTotalThis').value
    return {'Total Capex' : total_capex, 'Drilling Total': drilling_total, 'Completion Total': completion_total}

# ================== main code ==================

if __name__ == "__main__":
    browser = init_browser()
    login(browser)
    set_play(browser, selected_play)
    
    for drilling_day in drilling_days:
        set_drilling_days(browser, drilling_day)
        print(f'Drilling Days: {drilling_day} - {extract_cost(browser)}')
    
    for mod_factor in mod_factors:
        set_mod_factor(browser, mod_factor)
        print(f'Modification Factor: {mod_factor} - {extract_cost(browser)}')
    
    for drilling_day in drilling_days:
        set_drilling_days(browser, drilling_day)
        print(f'Drilling Days: {drilling_day} - {extract_cost(browser)}')
    
    for mod_factor in mod_factors:
        set_mod_factor(browser, mod_factor)
        print(f'Modification Factor: {mod_factor} - {extract_cost(browser)}')