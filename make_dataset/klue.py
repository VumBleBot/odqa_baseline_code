import json
import datasets
from copy import deepcopy
from datasets import DatasetDict, Dataset

def preprocessing_kluev2_to_kluev1(args):
    klue_v2_datasets = datasets.load_dataset("klue", "mrc")
    
    sub_dataset = {
        'id' : [],
        'title' : [],
        'context' : [],
        'question' : [],
        'answers': [],
        'document_id': [],
        '__index_level_0__' : []
    }
    
    klue_v1_dict = {
        "train": deepcopy(sub_dataset),
        "validation": deepcopy(sub_dataset)
    }
    
    wiki_pedia = {}
    
    document_id = 0
    
    for typ in ["train", "validation"]:
        for temp_dict in klue_v2_datasets[typ]:
            if temp_dict['source'] != 'wikipedia':
                continue
            
            klue_v1_dict[typ]['id'].append(temp_dict['guid']) 
            klue_v1_dict[typ]['title'].append(temp_dict['title']) 
            klue_v1_dict[typ]['context'].append(temp_dict['context'])
            klue_v1_dict[typ]['question'].append(temp_dict['question'])
            klue_v1_dict[typ]['answers'].append(temp_dict['answers'])
            klue_v1_dict[typ]['document_id'].append(document_id)
            klue_v1_dict[typ]['__index_level_0__'].append("0")
            
            wiki_pedia[str(document_id)] = {
                'url' : 'TODO',
                'text' : temp_dict['context'],
                'corpus_source' : '위키피디아',
                'domain' : None,
                'title' : temp_dict['title'],
                'author' : None,
                'html': None,
                'document_id': document_id,
                'id' : temp_dict['guid']
            }

            document_id += 1

    klue_v1_datasets = DatasetDict()
    klue_v1_datasets['train'] = Dataset.from_dict(klue_v1_dict['train'])
    klue_v1_datasets['validation'] = Dataset.from_dict(klue_v1_dict['validation'])
            
    return wiki_pedia, klue_v1_datasets

wiki_pedia, klue_v1_datasets = preprocessing_kluev2_to_kluev1(3)

with open("./input/data/wikipedia_documents.json", "w") as f:
    f.write(json.dumps(wiki_pedia, indent=4, ensure_ascii=False) + "\n")

klue_v1_datasets.save_to_disk("./input/data/train_dataset")
# klue_v1_datasets['validation'] = klue_v1_datasets['validation'].select(range(240))  # Klue V1 Dataset과 같은 Setting으로 변경