import pytest 
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service 
from webdriver_manager.chrome import ChromeDriverManager 
def test_open_site(): 
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install())) 
    driver.get('https://selenium.dev/') 
    driver.quit()