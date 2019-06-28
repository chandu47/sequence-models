import numpy as np
import re
import math
###Some paramters needed, Window size, epochs, batch size

##WE mat
lr=0.1
training_data=[]

def softmax(x): 
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum() 

def load_data_from_file():
	with open("sample-data.txt","r") as fp:
		print(type(fp))
		data=fp.read()
		data=data.lower()
	return data

def get_tokens_from_data(data):
	pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
	return pattern.findall(data)

def make_training_data(tokens,WINDOW_SIZE):
	###List of tuples(target,context)
	training_data_list=[]
	for i,token in enumerate(tokens):
		targets=[]
		for j in range(1,WINDOW_SIZE+1):
			if(i+j<len(tokens)):
				targets.append(tokens[i+j])
			if(i-j>-1):
				targets.append(tokens[i-j])
		training_data_list.append((token,targets))
	return training_data_list


def init_mats(vocab_size,embed_size):
	##embed mat should be embed_size x vocab_size
	embed_wt = np.random.randn(embed_size,vocab_size) * 0.01

	##embed_to_wt mat should be vocab_size x embed_size
	embed_to_word_wt = np.random.randn(vocab_size, embed_size ) * 0.01

	return embed_wt,embed_to_word_wt

def feed_forward(input_tup,embed_wt,embed_to_word_wt,vocab_size,embed_size):
	
	word_to_feed = token_to_id[input_tup]
	word_hc = np.zeros((vocab_size,1))
	word_hc[word_to_feed]=1


	hidden_layer = np.dot(embed_wt,word_hc)

	
	output_layer = np.dot(embed_to_word_wt, hidden_layer)

	y_soft=softmax(output_layer)

	return (y_soft,output_layer,hidden_layer)

def backprop(input_tup,embed_wt,embed_to_wt,vocab_size,embed_size,output_layer,h,eu):

	word_to_feed = token_to_id[input_tup]
	word_hc = np.zeros((vocab_size,1))
	word_hc[word_to_feed]=1

	dWhu = np.matmul(eu,np.transpose(h))
	#print(dWhu.shape)
	dWih = np.dot(word_hc,np.dot(np.transpose(eu),embed_to_wt))
	dWih=np.transpose(dWih)
	#print(dWih.shape)

	embed_wt -= lr*dWih
	embed_to_wt -= lr*dWhu

	return embed_wt,embed_to_wt

def train(epochs,embed_wt,embed_to_word_wt,vocab_size,embed_size):
	for i in range(epochs):
		loss=0
		for train_x,train_y in training_data:
			y_soft,out,h = feed_forward(train_x,embed_wt,embed_to_word_wt,vocab_size,embed_size)
			train_y_hc=[]
			for y in train_y:
				temp = token_to_id[y]
				temp_hc = np.zeros((vocab_size,1))
				temp_hc[temp]=1
				train_y_hc.append(temp_hc)	

			####calculating error
			EI = np.sum([np.subtract(y_soft, word) for word in train_y_hc], axis=0)
			#print(EI)

			##calculating loss
			loss += -np.sum([out[token_to_id[word]] for word in train_y]) + len(train_y) * np.log(np.sum(np.exp(out)))

			embed_wt,embed_to_word_wt =  backprop(train_x,embed_wt,embed_to_word_wt,vocab_size,embed_size,out,h,EI)

		print('EPOCH:',i, 'LOSS:', loss)
	print(embed_wt)


data=load_data_from_file()

tokens = get_tokens_from_data(data)

##Creating vocab

token_to_id = {}
id_to_token = {}
for i,token in enumerate(set(tokens)):
	token_to_id[token]=i
	id_to_token[i]=token



print(len(token_to_id))

vocab_size=len(token_to_id)

##using a window size of two for skip-gram model
training_data = make_training_data(tokens,2)

print(training_data[0])
#print(training_data)

##Taking the embedding size of 64

embed_size=64

##initializing the weights matrix
embed_wt,embed_to_word_wt = init_mats(vocab_size,embed_size)
print(embed_wt)
print(embed_wt.shape)

train(200,embed_wt,embed_to_word_wt,vocab_size,embed_size)
###setting learning rate 0.1


 
