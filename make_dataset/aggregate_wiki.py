import json
import pandas as pd
from tools import get_args

def aggregate_wiki(args):
    print("start aggregating wiki docs")
    json_path = '/opt/ml/input/data/wikipedia_documents.json'
    with open(json_path) as wiki_file:
        wiki_dict = json.load(wiki_file)

    df = pd.DataFrame.from_dict(wiki_dict, orient='index')
    title_dict = dict.fromkeys(set(df['title']))

    for i in range(len(df)):
        if title_dict[df["title"][i]] is None:
            temp = {
                'text': df['text'][i],
                'document_id': [df['document_id'][i]]
            }
            title_dict[df["title"][i]] = temp
        else:
            title_dict[df["title"][i]]['text'] += '#' + df['text'][i]
            title_dict[df["title"][i]]['document_id'].append(df['document_id'][i])

    new_df = pd.DataFrame(title_dict)
    new_df = new_df.transpose()
    new_df = new_df.reset_index()
    new_df.columns = ['title', 'text', 'document_id']

    new_df.to_json('/opt/ml/input/data/wikipedia_documents_agg.json', orient='index')
    print("wiki docs saved")

def main(args):
    aggregate_wiki(args)

if __name__ == "__main__":
    args = get_args()
    main(args)