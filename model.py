import numpy as np
from keras.layers import LSTM,Input,Dense,Attention,Masking
from keras.models import Model
import random
from DataLoader import *



class MachineTranslation:
    
    def __init__(self):
        
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()
        # self.english_characters = None
        # self.french_characters = None
        self.input_index = {}
        self.target_index = {}
        self.reverse_input_index = {}
        self.reverse_target_index = {}
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
        self.max_english_sentence_length = 0
        self.max_french_sentence_length = 0
        self.num_english_sentences = 0
        self.num_target_sentences = 0
        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None
        self.model = None

    def function_1(self,file):
        lines = file.split('\n')
        for line in lines[:min(10000,len(lines) - 1)]:
            input_text,target_text = line.split('\t')
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in self.input_characters:
                    self.input_characters.add(char)
            for char in target_text:
                if char not in self.target_characters:
                    self.target_characters.add(char)

        self.input_characters = sorted(self.input_characters)
        self.target_characters = sorted(self.target_characters)

        self.max_english_sentence_length = max([len(sent) for sent in self.input_texts])
        self.max_french_sentence_length = max([len(sent) for sent in self.target_texts])

        self.num_english_sentences = len(self.input_texts)
        self.num_target_sentences = len(self.target_texts)

        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)

    def create_dict(self):
        self.input_index = dict((char,i) for i,char in enumerate(self.input_characters))
        self.target_index = dict((char,i) for i,char in enumerate(self.target_characters))

        self.reverse_input_index = dict((i,char) for char,i in self.input_index.items())
        self.reverse_target_index = dict((i,char) for char,i in self.target_index.items())


    def create_dataset(self):
        self.encoder_input_data = np.zeros((self.num_english_sentences,self.max_english_sentence_length,self.num_encoder_tokens),dtype='float32')
        self.decoder_input_data = np.zeros((self.num_target_sentences,self.max_french_sentence_length,self.num_decoder_tokens),dtype='float32')
        self.decoder_target_data = np.zeros((self.num_target_sentences,self.max_french_sentence_length,self.num_decoder_tokens),dtype='float32')

    def one_hot_encoding(self):
        for i,(input_text,target_text) in enumerate(zip(self.input_texts,self.target_texts)):
            for t,char in enumerate(input_text):
                self.encoder_input_data[i,t,self.input_index[char]] = 1
            self.encoder_input_data[i,t+1:,self.input_index[' ']] = 1

            for t,char in enumerate(target_text):
                self.decoder_input_data[i,t,self.target_index[char]] = 1
                if t > 1:
                    self.decoder_target_data[i,t-1,self.target_index[char]] = 1
            self.decoder_input_data[i,t+1:,self.target_index[' ']] = 1
            self.decoder_target_data[i,t:,self.target_index[' ']] = 1


    def build_model(self):

        # encoder
        encoder_inputs = Input(shape=(None,self.num_encoder_tokens),name="encoder_input_layer")
        encoder_outputs, state_h, state_c = LSTM(256, return_state=True)(encoder_inputs)
        encoder_states = [state_h, state_c]

        # decoder model
        decoder_inputs = Input(shape=(None,self.num_decoder_tokens),name="decoder_input_layer")
        decoder,h,c = LSTM(256,return_sequences=True,return_state=True,name="decoder")(decoder_inputs,initial_state=encoder_states)
        decoder_outputs = Dense(units=self.num_decoder_tokens,activation='softmax')(decoder)

        self.model = Model([encoder_inputs,decoder_inputs],[decoder_outputs])


    def fit_data(self,batch_size=64,epochs=100,validation_split=0.1):
        combined = list(zip(self.encoder_input_data,self.decoder_input_data,self.decoder_target_data))
        random.shuffle(combined)
        # Split the shuffled data back into separate arrays
        encoder_input_data, decoder_input_data, decoder_output_data = zip(*combined)
        # Convert the arrays back to NumPy arrays if needed
        self.encoder_input_data = np.array(encoder_input_data)
        self.decoder_input_data = np.array(decoder_input_data)
        self.decoder_output_data = np.array(decoder_output_data)

        if self.model.optimizer is not None and self.model.loss is not None:
            print("Model has been Compiled Successfully...")
            self.model.fit([self.encoder_input_data,self.decoder_input_data],self.decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=validation_split,verbose=1)
        else:
            print("Model has not been compiled please compile the model.")


    def train(self,optimizer="rmsprop",loss="categorical_crossentropy",metrics=['accuracy']):
        self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        self.fit_data()


    def save_model(self,destination_path):
        self.model.save(destination_path)


    def load_data(self,data_path):
        data = DataLoader.load(data_path)
        self.function_1(data)
        self.create_dict()
        self.create_dataset()
        self.one_hot_encoding()