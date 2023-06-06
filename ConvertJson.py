import json
import csv

instruction = "You will answer questions about 20000 leagues under the sea "

# Load the data from the JSON file
with open('C:\\Users\\jeffr\\Downloads\\data.json', 'r') as f:
    responses = json.load(f)

# Open the CSV file and write the data to it
with open('responses.csv', 'w', newline='') as csvfile:
    fieldnames = ['question', 'answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for response in responses['responses']:
        if 'question' in response and 'answer' in response:  # To make sure both keys exist
            writer.writerow({'question': instruction + response['question'], 'answer': response['answer']})

