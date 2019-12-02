import pandas as pd
import pickle


pd.options.display.max_colwidth = 1000

###### Data Extraction ######

excel_file = 'Dataset/1830_Training.xlsx'
sheet_one = 'UK'
sheet_two ='Undefined'

train_df1 = pd.read_excel(excel_file, sheet_name =sheet_one)
train_df2 = pd.read_excel(excel_file, sheet_name =sheet_two)


train_df = pd.concat([train_df1, train_df2])
train_df['complete'] = train_df['Context-left'] + train_df['Term'] + ' ' + train_df['Context-Right']
train_df['complete'] = train_df['complete'].str.replace('- ', '')
train_df['complete'] = train_df['complete'].str.replace(' \'s', '\'s')

#print(train_df['complete'])


###### Convert Data to Spacy Training Format ######

train_format = train_df[['complete', 'Term']]


TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]



train_format['check'] = ""


train_format = train_format[pd.notnull(train_format['complete'])]

train_list = []


for index, row in train_format.iterrows():
    row['check'] = str(row['complete']).find(row['Term'])
    one = row['complete']
    indexList = (row['check'], row['check'] + len(row['Term']), "GPE")
    two = {'entities': [indexList]}
    train_list.append((one,two))

#print(train_list)

with open('Dataset/outfile.txt', 'wb') as fp:
   pickle.dump(train_list, fp)


print(train_list)
















