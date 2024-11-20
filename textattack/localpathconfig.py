
import os

class LocalPathConfig:

    current_dir = os.path.dirname(os.path.realpath(__file__))
    #bert-base-uncased
    BERT_BASE_UNCASED =  os.path.abspath(os.path.join(current_dir, '../../google/bert-base-uncased'))

    UNIVERSAL_SENTENCE_ENCODER = os.path.abspath(os.path.join(current_dir, '../../google/universal-sentence-encoder'))

    UNIVERSAL_SENTENCE_ENCODER_LARGE = os.path.abspath(os.path.join(current_dir, '../../google/universal-sentence-encoder-large'))
    
    WORDENBEDDINGS=os.path.abspath(os.path.join(current_dir, '../../word_embeddings'))
    
    
    
    
if __name__ == "__main__":
    print("BERT_BASE_UNCASED:", LocalPathConfig.BERT_BASE_UNCASED)
    print("UNIVERSAL_SENTENCE_ENCODER:", LocalPathConfig.UNIVERSAL_SENTENCE_ENCODER)
    print("UNIVERSAL_SENTENCE_ENCODER_LARGE:", LocalPathConfig.UNIVERSAL_SENTENCE_ENCODER_LARGE)
    print("WORDENBEDDINGS:", LocalPathConfig.WORDENBEDDINGS)
