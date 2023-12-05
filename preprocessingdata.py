import json

# Path to the JSON file
file_path = 'Dataset for Detection of Cyber-Trolls.json'
#path in colab
#file_path = '/content/Dataset for Detection of Cyber-Trolls.json'
# Reading the file
with open(file_path, 'r') as file:
    data = file.readlines()

# Convert the JSON strings to dictionaries and store them in a list
tweets = [json.loads(line) for line in data]
