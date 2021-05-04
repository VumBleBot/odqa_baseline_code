from transformers import BertModel, BertPreTrainedModel, BertConfig, AutoTokenizer

class BertEncoder(BertPreTrainedModel):
    def __init__(self,config):
        super(BertEncoder,self).__init__(config)
        
        self.bert=BertModel(config)
        self.init_weights()
        
    def forward(self,input_ids,attention_mask=None, token_type_ids=None):
        outputs=self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output=outputs[1] # embedding 가져오기
        return pooled_output
