import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from fake_useragent import UserAgent
from selenium.webdriver.support import expected_conditions as EC
import json, os
# from extract_examples import Extract_Examples

with open("Files/ac2w0.00.json") as f:
    data = [json.loads(line) for line in f]

metrics = {}
for generated_text in data[0]["generation_complete"]:
   while True:
      try:
         op = webdriver.ChromeOptions()
         op.add_argument(f"user-agent={UserAgent.random}")
         op.add_argument("user-data-dir=./")
         op.add_experimental_option("detach", True)
         op.add_experimental_option("excludeSwitches", ["enable-logging"])

         driver = uc.Chrome(chrome_options=op)
         PATH = "chromedriver-linux64/chromedriver"
         driver.get('https://chat.openai.com/')

         inputElements = driver.find_elements(By.TAG_NAME, "textarea")

         prompt = "I am given a task of generating lexically constrained sentences which are meaningful. The lexical constraints are simply certain words which must be present in the generated sentence in any tense form."
         prompt = prompt + f" Given constraints: stand, look, field. Following are examples of 'good' sentences that satisfy the constraints while also being meaningful: 1.The player stood in the field looking at the batter, 2.The coach stands along the field, looking at the goalkeeper, 3.I stood and looked across the field, peacefully, 4.Someone stands, looking around the empty field. Based on these suggestions, please rate the following generated sentence on a scale of 1-5 for each of the metrics: Constraint Satisfaction, Fluency, Meaningfulness. Constraint satisfaction is a metric which checks if all of the constraint words are present in the generated sentence. If none of the constraint words are present, give 0. If one of them is present give 1, if two are present give 2, if all three are present give 3. Give no marks more than 3 for Constraint Satisfaction. Fluency is a general metric to judge the syntactical and grammatical correctness. If the fluency is very poor, give 0. If fluency is very good, give 5. Meaningfulness measures if the generated sentence makes any sense. If the generated sentence is not meaningful give 0, if it makes complete sense give 5. Please provide a score for each metric in the format: Constraint Satisfaction: <score>, Fluency: <score>, Meaningfulness: <score>. Please do not include any other information in your response other than the scores for each metric. Generated sentence: {generated_text}\t\t"
         # Write the prompt to the text area
         inputElements[0].send_keys(prompt)
         sleep(5)
         inputElements[0].send_keys(Keys.RETURN)
         sleep(3)
         inputElements = driver.find_elements(By.TAG_NAME, "p")
         sleep(2)
         results = []
         for element in inputElements:
            results.append(element.text)
         print(generated_text)
         print(results)
         metrics[generated_text] = results[-2]
         driver.quit()
         break
      except:
         driver.quit()
         continue

with open("Files/ac2w0.00_metrics.json", "w") as f:
    json.dump(metrics, f)
