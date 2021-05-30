# Reader 구조

**현재 Reader의 경우 backbone으로써 BERT 기반 모델(bert, distillbert, albert, xlm-roberta 등)과 ELECTRA 기반 모델만 활용할 수 있도록 구현되어있습니다.**  
    
만약 그 외의 모델을 백본으로 활용하시고 싶으시다면 `reader/custom_reader.py`의 `CustomModel` 클래스에서 `forward()` 메소드를 해당 백본에 맞도록 변경해주셔야합니다. 또한 `utils/prepare.py`의 `get_reader()` 메소드에서 해당 모델을 정상적으로 가져올 수 있도록 코드를 수정해주셔야합니다.  

## Custom head
Reader는 pretrained-backbone을 활용한다는 가정 하에 설계되었으며, 따라서 백본 단에서 별도의 수정을 하지 않았고 head 부분만 커스터마이징하였습니다.  
  
**만약 직접 커스터마이징한 헤드를 추가하고 싶다면 아래와 같은 프로세스를 따라주세요.**
1. `reader/custom_head.py`에 원하는 head를 모델링합니다.
2. `reader/custom_reader.py`에서 추가한 head를 import하고 `READER_HEAD` 딕셔너리에 해당 헤드를 alias와 함께 추가합니다.
3. 전략 config 파일에서 해당 alias를 `reader_name`으로 줍니다.
  
아래는 기본적으로 제공되고있는 헤드와 설계구조입니다.

### DPR
가장 기본적인 fully connected head입니다.

### LSTM
Transformer model의 output을 다시 LSTM에 넣은 이후 fully connected layer를 통과시켜 최종 output을 도출해내는 헤드입니다.

### CNN
out channel 2의 kernel size 1, 3, 5인 총 3개의 1D CNN을 병렬적으로 통과시킨 후 그 결과를 모두 더하여 최종 output을 도출해내는 헤드입니다.

### CCNN
앞선 CNN모델과 같은 구조이나, out channel을 256으로 설정하고 마지막으로 이들을 더하지 않고 concatenation하여 최종적으로 fully connected layer를 통과시키는 모델입니다.

### CCNN_v2
앞선 CCNN 모델에 dropout을 추가한 모델입니다. default는 0.5이며 필요하실 경우 조정하고 사용하시면 됩니다.

### CNN_LSTM
LSTM 모델과 CCNN_v2 모델을 결합한 형태의 모델입니다.
먼저 LSTM을 통과시킨 후, 이에 대한 output을 CCNN_v2 모델에 통과시킵니다. default dorpout 수치는 0.3입니다.  

### CCNN_EM
CCNN_v2 구조에 Exact Match(EM) token을 추가적인 feature로 활용하는 모델입니다. EM token에 대한 자세한 설명은 [Wrap-up report](https://hackmd.io/@9NfvP9AZQL2Psilxs3oNBA/SyH-EkVt_)를 참고해주세요.

### CCNN_LSTM_EM
CNN_LSTM 구조에 EM token을 추가적인 feature로 활용하는 모델입니다.  
  