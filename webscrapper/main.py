'''
web scrapping sample code
'''

from splinter import Browser
from splinter.exceptions import ElementDoesNotExist
from bs4 import BeautifulSoup as bs
import numpy as np
import pandas as pd
import time

# ================== input values ==================

drilling_days = 20

# ================== functions ==================

def init_browser():
    executable_path = {'executable_path': 'chromedriver.exe'}
    return Browser('chrome', **executable_path, headless=False)

def scrape_info(drilling_days: int) -> dict:
    browser = init_browser()

    # TODO: login if needed
    costanalyst_url = 'http://sna-powertrain-orange.ihsmarkit.com/#/unconv/cost/costAnalysisGrid'
    welldata_url = 'http://sna-powertrain-orange.ihsmarkit.com/#/asset/2554/welldata'
    costsummary_url = 'http://sna-powertrain-orange.ihsmarkit.com/#/asset/2554/costsummary/summary'

    # visit cost analyst grid
    browser.visit(costanalyst_url)

    # TODO: select Play/Subplay

    # scrape page into Soup
    html = browser.html
    soup = bs(html, "html.parser")
    

    # # wait for data to load
    # time.sleep(3) 

    
        

    return {'Drilling Total' : 0.0}

# ================== main code ==================

if __name__ == "__main__":
    data = scrape_info(drilling_days)
    # df = pd.DataFrame(data, columns =['code', 'link', 'price', 'bed', 'bath', 'sqft'])
    # df.to_csv("zillow.csv")