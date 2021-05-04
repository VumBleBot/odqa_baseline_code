from base_retrieval import Retrieval
from encoders import BertEncoder
from transformers import BertModel, BertPreTrainedModel, BertConfig, AutoTokenizer
import pickle
import numpy as np

class DenseRetrieval(Retrieval): # 
    def __init__(self, args, tokenize_fn): # (Question) 이렇게 하면, Retrieval의 args로 들어가는 것 맞음? 
        # 궁금함 - super init 해도 애초에 인자 받는 애는 
        super().__init__(args,tokenize_fn) # 넘겨받은 파라미터 그대로 initialize
        self.name = 'dpr' # eg. tf-idf, bm25, dpr
        self.embedding_arc = BertEncoder.from_pretrained(args.model.model_name_or_path).cuda() # TfidfVectorizer or BertEncoder
        self.tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path) # 고쳐야함. 넘겨받은 tokenizer로 
        self.indexer = None 

    # override
    def exec_embedding(self): # tf-idf의 경우는 vecotrizer, 다른 애는 encoder. 부모 클래스에 접근하기 전에 overriding됨
        """
            (Function) override parents' func
            (Return) embedding vector, vectorizer or encoder
        """
        ##############################################################
        # 여기에 encoder 학습해서 pth 저장하고 클래스 변수에 로딩해야 한다. 지금은 그냥 불러옴 #######
        # 일단, pth 있다고 가정. 파일 있는지 테스트하고 가져와야. 없으면 학습하고 반환해야 . 근데 pth를 피클로 저장함? 
        # 아니면, 학습 후 pth 저장하는 로직까지. 일단 그냥 넘기는 방향
        model_dict=torch.load("../../input/data/encoder.pth") # [TO-DO] fine-tuning
        self.embedding_arc.load_state_dict(model_dict['p_encoder'])
        ##############################################################
        with torch.no_grad():
            self.embedding_arc.eval()
            p_embs=[]
            for p in self.contexts: # self.contexts가 지금 무슨 자료형인지는 모른다. 
                p=self.tokenizer(p,padding="max_length",truncation=True, return_tensors='pt').to('cuda')
                p_emb=self.embedding_arc(**p).to('cpu').numpy()
                p_embs.append(p_emb)
        return p_embs, None

    # override
    def get_relevant_doc(self, queries, k=1):
        # query 를 임베딩. encoder를 사용 
        q_encoder = BertEncoder.from_pretrained(args.model.model_name_or_path).cuda()
        model_dict=torch.load("../../input/data/encoder.pth") # [TO-DO] fine-tuning
        q_encoder.load_state_dict(model_dict['q_encoder'])
        with torch.no_grad():
            q_encoder.eval()
            q_seqs_val=self.tokenizer(queries, padding="max_length",truncation=True,return_tensors='pt').to('cuda') # get_relevant_doc 여러번 하는 것보다 [query,query2..] 으로 늘려주는게 빠름. 즉, 인풋으로 받는 [query]를 querylist로 받으면 될 듯
            q_emb=q_encoder(**q_seqs_val).to('cpu') # (num_query, emb_dim)
            
        dot_prod_scores=torch.matmul(q_emb,torch.transpose(p_embs,0,1)) # 각각 임베딩 dot product해서 score구하기
        rank=torch.argsort(dot_prod_scores,dim=1,descending=True).squeeze()
        # 똑같이 리스트 형태인지 확인하기. 지금은 bulk용 아닌듯 
        return dot_prod_scores.squeeze(), rank[:k]

