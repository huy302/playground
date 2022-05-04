'''
Robot service class
'''
from splinter import Browser
from selenium.common.exceptions import ElementClickInterceptedException
import time

import secrets

class RobotService:
    def __init__(self) -> None:
        self.__browser = self.__init_browser()
        self.__login()
        self.__set_play('Bakken Shale') # TODO: hard coded for now

    def __init_browser(self):
        executable_path = {'executable_path': 'chromedriver.exe'}
        return Browser('chrome', **executable_path, headless=False)

    def __login(self) -> None:
        '''
        System login
        '''
        login_url = 'http://spowertrain-orange.ihsmarkit.com/Account/Login'
        costanalyst_url = 'http://spowertrain-orange.ihsmarkit.com/Account/RedirectTo?returnUrl=http%3A%2F%2Fsna-powertrain-orange.ihsmarkit.com%2F'

        # login
        self.__browser.visit(login_url)
        self.__browser.find_by_id('UserName').fill(secrets.login_creds['username'])
        self.__browser.find_by_id('Password').fill(secrets.login_creds['password'])
        self.__browser.execute_script("document.getElementsByTagName('form')[1].submit()")
        time.sleep(2) # wait for content to load
        # visit cost analysis page
        self.__browser.visit(costanalyst_url)
        time.sleep(2) # wait for content to load
        # click on 'Cost Analysis Grid' button
        self.__browser.find_by_text(' Cost Analysis Grid').click()
        time.sleep(2) # wait for content to load
    
    def __set_play(self, play: str) -> None:
        '''
        Set Play/Subplay and navigate to Well Data panel
        '''
        # select Play/Subplay
        play_dropdown = self.__browser.find_by_css('div[class="cost-analysis-grid-header-play-subplay"]').first.find_by_tag('select').first
        play_dropdown.find_by_text(play).click()
        time.sleep(2) # wait for content to load
        # click on Well Data pencil button
        self.__browser.find_by_css('i[class="fa fa-pencil-square fa-2x"]').first.click()
        time.sleep(2) # wait for content to load

    def set_drilling_days(self, drilling_days: int) -> dict:
        '''
        Set drilling days and save
        '''
        # go to Well Data tab
        self.__browser.find_by_text('Well Data').click()
        # set drilling days and hit Save
        self.__browser.find_by_id('DrillingDays').fill(drilling_days)
        self.__browser.find_by_text('Save').click()
        time.sleep(1) # wait for content to save
        alert = self.__browser.get_alert()
        alert.accept()
        return self.__extract_cost()
    
    def set_mod_factor(self, mod_factor: float) -> dict:
        '''
        Set modification factor and save
        '''
        # go to Drilling Cost tab
        self.__browser.find_by_text('Drilling Cost').first.click()
        # edit Permit and Survey cost sheet
        self.__browser.find_by_text('Permit and Survey').first.find_by_xpath('..').first.find_by_css('i[class="fa fa-pencil-square-o fa-2x"]').first.click()
        # set modification factor, comments and hit Save
        self.__browser.find_by_id('ModificationFactor').fill(str(mod_factor))
        if mod_factor != 1:
            self.__browser.find_by_id('modCommentsForEdit').fill('Automated service')
        # try to hit Save, if Save is disabled then hit Cancel
        try:
            self.__browser.find_by_text('Save').click()
        except ElementClickInterceptedException:
            self.__browser.find_by_text('Cancel').click()
        time.sleep(1) # wait for content to save
        return self.__extract_cost()
    
    def __extract_cost(self) -> dict:
        '''
        Click on Cost Summary and return results
        '''
        self.__browser.find_by_text('Cost Summary').click()
        total_capex = self.__browser.find_by_id('curCapexTotalThis').value
        drilling_total = self.__browser.find_by_id('curDrillTotalThis').value
        completion_total = self.__browser.find_by_id('curCompTotalThis').value
        return {'Total Capex' : total_capex, 'Drilling Total': drilling_total, 'Completion Total': completion_total}