import numpy as np
import math
from scipy import special

H = 100 ##hidden layer neurons, should be same as C state of lstm cell?


def sigmoid(x):
	#print(x)
	return 1 / (1 + np.exp(-x))

def dsigmoid(x):
	f=sigmoid(x)
	return f*(1.-f)

def tanh(x):
	return np.tanh(x)

def dtanh(x):
	temp=1.-(np.tanh(x))**2
	return temp
def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


class LSTM:
	
	def __init__(self,h,d,seq_len):
		self.H=h ##hidden layer dim
		self.seq_length=seq_len ##seq len
		self.D=d ###input word dim (or vocab len)
		self.Z=self.H+self.D ###Concated length of input and hidden layer length
		self.model_params()

	def model_params(self):
		##Weights
		self.Wf=np.random.randn(self.Z,self.H) / np.sqrt(self.Z / 2.)
		self.Wi=np.random.randn(self.Z,self.H) / np.sqrt(self.Z / 2.)
		self.Wc=np.random.randn(self.Z,self.H) / np.sqrt(self.Z / 2.)
		self.Wo=np.random.randn(self.Z,self.H) / np.sqrt(self.Z / 2.)
		self.Wy=np.random.randn(self.H,self.D) / np.sqrt(self.D / 2.)

		###Biases
		self.bf=np.zeros((1,self.H))
		self.bi=np.zeros((1,self.H))
		self.bc=np.zeros((1,self.H))
		self.bo=np.zeros((1,self.H))
		self.by=np.zeros((1,self.D))

		##Derivates for back_prop
		self.reset_grads()

	def reset_grads(self):
		self.dWf=np.zeros_like(self.Wf)
		self.dWi=np.zeros_like(self.Wi)
		self.dWc=np.zeros_like(self.Wc)
		self.dWo=np.zeros_like(self.Wo)
		self.dWy=np.zeros_like(self.Wy)

		self.dbf=np.zeros_like(self.bf)
		self.dbi=np.zeros_like(self.bi)
		self.dbc=np.zeros_like(self.bc)
		self.dbo=np.zeros_like(self.bo)
		self.dby=np.zeros_like(self.by)
	
	def update_grads(self,grads_wts,grads_biases):
		dWf,dWi,dWc,dWo,dWy = grads_wts
		dbf,dbi,dbc,dbo,dby = grads_biases

		self.dWf+=dWf
		self.dWi+=dWi
		self.dWc+=dWc
		self.dWo+=dWo
		self.dWy+=dWy

		self.dbf+=dbf
		self.dbi+=dbi
		self.dbc+=dbc
		self.dbo+=dbo
		self.dby+=dby

	def update_weights(self,lr):
		self.Wf-=lr*self.dWf/self.seq_length
		self.Wi-=lr*self.dWi/self.seq_length
		self.Wc-=lr*self.dWc/self.seq_length
		self.Wo-=lr*self.dWo/self.seq_length
		self.Wy-=lr*self.dWy/self.seq_length

		###Biases
		self.bf-=lr*self.dbf/self.seq_length
		self.bi-=lr*self.dbi/self.seq_length
		self.bc-=lr*self.dbc/self.seq_length
		self.bo-=lr*self.dbo/self.seq_length
		self.by-=lr*self.dby/self.seq_length

	def forward_pass(self,X,state):
		##state saves the previous state conditions(c_old and h_old)
		h_old,c_old=state
		X_hc=np.zeros((1,self.D))
		X_hc[0][X]=1.

		###Stacking the two vectors
		X = np.column_stack((h_old,X_hc))
		
		###Computing gate values
		hf = sigmoid(X@self.Wf+self.bf)
		hi = sigmoid(X@self.Wi+self.bi)
		hc = tanh(X@self.Wc+self.bc)
		ho = sigmoid(X@self.Wo+self.bo)

		c=hf*c_old+hi*hc
		h=ho*tanh(c)

		y=h@self.Wy+self.by

		out=softmax(y)

		state=h,c
		cached_data=(hf,hi,hc,ho,c,h,y,c_old,X)
		return cached_data,out,state

	def backprop(self,pred,y_gt,d_next,cached_data):
		hf,hi,hc,ho,c,h,y,c_old,X=cached_data

		h_next,c_next=d_next

		dy=pred.copy()

		dy[0][y_gt]-=1.

		#Weights for out
		dWy=h.T@dy
		dby=dy

		dh=dy@self.Wy.T + h_next

		###gradeient for h0
		dho=dh*tanh(c)
		dho=dsigmoid(ho)*dho

		# Gradient for c in h = ho * tanh(c), note we're adding dc_next here
		dc=dh*ho*dtanh(c)
		dc+=c_next

		# Gradient for hf in c = hf * c_old + hi * hc
		dhf=dc*c_old
		dhf=dsigmoid(hf)*dhf

		# Gradient for hi in c = hf * c_old + hi * hc
		dhi=dc*hc
		dhi=dsigmoid(hi)*dhi

		# Gradient for hc in c = hf * c_old + hi * hc
		dhc=dc*hi
		dhc=dtanh(hc)*dhc

		# Gate gradients, just a normal fully connected layer gradient
		##forget gate weights
		dWf=X.T@dhf
		dbf=dhf
		dXf=dhf@self.Wf.T

		###i gate values
		dWi=X.T@dhi
		dbi=dhi
		dXi=dhi@self.Wi.T

		##activation gate weights
		dWc=X.T@dhc
		dbc=dhc
		dXc=dhc@self.Wc.T

		##output gate weigths
		dWo=X.T@dho
		dbo=dho
		dXo=dho@self.Wo.T

		##Change in input
		dX=dXf+dXo+dXi+dXc

		dh_next = dX[:, :self.H]
		dc_next=hf*dc

		d_next=dh_next,dc_next

		ders_wts=(dWf,dWi,dWc,dWo,dWy)
		ders_biases=(dbf,dbi,dbc,dbo,dby)
		#for dparam in ders_biases:
		#	np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		#for dparam in ders_wts:
		#	np.clip(dparam, -5, 5, out=dparam)
		return d_next,ders_wts,ders_biases

	def train_step(self,inputs,targets,state):
		preds=[]
		caches=[]
		loss=0.
		state=state

		for x,y_true in zip(inputs,targets):
			cached_data,out,state=self.forward_pass(inputs,state)
			preds.append(out)
			caches.append(cached_data)
			loss+= -np.log(out[0][y_true])
		loss/=self.seq_length

		d_next = (np.zeros_like(H), np.zeros_like(H))
		self.reset_grads()
		for prob, y_true, cache in reversed(list(zip(preds, targets, caches))):
			d_next, grads_wts, grads_biases = self.backprop(prob, y_true, d_next, cache)
			self.update_grads(grads_wts,grads_biases)
		
		return 	loss,state


	def prepare_data_and_train(self,epochs,lr,data,char_to_ix,ix_to_char):
		n,p=0,0

		for epoch in range(epochs):
			if p+self.seq_length+1 >= len(data) or n == 0: 
				state = (np.zeros((1,self.H)),np.zeros((1,self.H))) # reset RNN memory
				p = 0 # go from start of data
			inputs = [char_to_ix[ch] for ch in data[p:p+self.seq_length]]
			targets = [char_to_ix[ch] for ch in data[p+1:p+self.seq_length+1]]
			
			loss , state = self.train_step(inputs,targets,state)
			
			self.update_weights(lr)
			if(epoch%100==0):
				print([ix_to_char[i] for i in inputs])
				print([ix_to_char[i] for i in targets])
				print("Epoch " + str(epoch) + ", Loss: "+ str(loss))
				print("Sample: ")
				sample_ix = self.produce_samples(inputs[0],200,state)
				txt = ''.join(ix_to_char[ix] for ix in sample_ix)
				print('----\n %s \n----' % (txt, ))

			n+=1
			p+=self.seq_length

	def produce_samples(self,seed,seqLen,state):
		state = state
		ixes=[]
		ixes.append(seed)
		for t in range(seqLen):
			_,p,state=self.forward_pass(seed,state)
			ix = np.random.choice(range(self.D), p=p.ravel())
			ixes.append(ix)
		return ixes


# data I/O
data = open('input.txt', 'r').read() 
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.'% (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
print(ix_to_char)
print(chars)
lstm_model=LSTM(H,vocab_size,30)
##learning rate
lr=0.1
lstm_model.prepare_data_and_train(50000,lr,data,char_to_ix,ix_to_char)
