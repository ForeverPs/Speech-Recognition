import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import wave
import librosa
import numpy as np
import librosa.display
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras.models import load_model
from keras.utils import to_categorical
import matplotlib.pylab as plt
from pyaudio import PyAudio,paInt16
from fastdtw import fastdtw
from collections import Counter
from python_speech_features.base import mfcc
from scipy import interpolate
from scipy.spatial.distance import euclidean

class FREQ(object):
	def __init__(self):
		self.len_frame=0.04
		self.ratio=0.3
		self.len_CNN=35000
		self.num_samples=1980
		self.test_num=190
		self.n_mfcc=20
		self.time=3
		self.num_net=6
		self.model=[]
		for i in range(self.num_net):
			self.model.append(load_model('cnn'+str(i)+'.h5'))
		self.framerate=44100
		self.sampwidth=2
		self.channels=1
		self.chunk=1024
		self.frames_per_buffer=2000
		#store 2000 points each time
		self.len_points=self.len_frame*self.framerate

	def save(self,filename,data):
		wf=wave.open(filename,'wb')
		wf.setnchannels(self.channels)
		wf.setsampwidth(self.sampwidth)
		wf.setframerate(self.framerate)
		wf.writeframes(b"".join(data))
		wf.close()

	def record(self):
		pa=PyAudio()
		storename='C:/Users/peisen/Desktop/temp/temp.wav'
		stream=pa.open(format=paInt16,channels=self.channels,rate=self.framerate,input=True,frames_per_buffer=self.frames_per_buffer)
		my_buffer=[]
		count=0
		print('\nRecording: ')
		while count<23*self.time:
			string_audio_data=stream.read(self.frames_per_buffer)
			my_buffer.append(string_audio_data)
			count+=1
			if count%10==0:
				print('.....')
		print('Finished\n')
		self.save(storename,my_buffer)
		stream.close()

	def change_freq(self,switch=True):
		if switch:
			filepath='C:/Users/peisen/Desktop/vedio/'
			target='C:/Users/peisen/Desktop/new_vedio/'
		else:
			filepath='C:/Users/peisen/Desktop/test_vedio/'
			target='C:/Users/peisen/Desktop/new_test_vedio/'
		print('\n'+18*'*'+'\n'+'Changing Frequency'+'\n'+18*'*')
		print('  wait...')
		for j in range(10):
			temp_filepath=filepath+str(j)+'/'
			new_path=target+str(j)+'/'
			filename=os.listdir(temp_filepath)
			for i in range(len(filename)):
				name=temp_filepath+filename[i]
				new_name=new_path+filename[i]
				y,sr = librosa.load(name, sr=None)
#				plt.figure()
#				plt.plot([x for x in range(y.shape[0])],y)
#				plt.show()
				new_y= librosa.resample(y,sr,self.framerate)
#				plt.figure()
#				plt.plot([x for x in range(new_y.shape[0])],new_y)
#				plt.show()
				librosa.output.write_wav(new_name,new_y,self.framerate)
		print('  OK\n')

	def play(self,name):
		wf=wave.open(name,'rb')
		params=wf.getparams()
		nchannels,sampwidth,framerate,nframes=params[:4]
		p=PyAudio()
		stream=p.open(format=p.get_format_from_width(sampwidth),
						channels=nchannels,rate=framerate,output=True)
		while True:
			data=wf.readframes(self.chunk)
			if data==b"":
				break
			stream.write(data)
		stream.close()
		p.terminate()

	def pre_feature_dtw(self,switch=True):
		if switch:
			filepath='C:/Users/peisen/Desktop/new_vedio/'
			label_name='train'
		else:
			filepath='C:/Users/peisen/Desktop/new_test_vedio/'
			label_name='test'
		print('\n'+15*'*'+'\n'+'Processing Data'+'\n'+15*'*')
		print('  wait...')
		labels=[]
		for j in range(10):
			temp_filepath=filepath+str(j)+'/'
			filename=os.listdir(temp_filepath)
			filename.sort(key=lambda x:int(x[:-4]))
			for i in range(len(filename)):
				name=temp_filepath+filename[i]
				orig=self.mel(name)
				labels.append(j)
				if switch:
					np.save('C:/Users/peisen/Desktop/orig_mel/'+str(1000*j+i)+'.npy',orig)
				else:
					np.save('C:/Users/peisen/Desktop/test_mel/'+str(1000*j+i)+'.npy',orig)
		np.save('C:/Users/peisen/Desktop/'+label_name+'.npy',labels)
		print('  OK\n')

	def pre_feature_cnn(self,switch=True):
		if switch:
			filepath='C:/Users/peisen/Desktop/new_vedio/'
			label_name='cnn_train'
		else:
			filepath='C:/Users/peisen/Desktop/new_test_vedio/'
			label_name='cnn_test'
		print('\n'+15*'*'+'\n'+'Processing Data'+'\n'+15*'*')
		print('  wait...')
		labels=[]
		for j in range(10):
			temp_filepath=filepath+str(j)+'/'
			filename=os.listdir(temp_filepath)
			filename.sort(key=lambda x:int(x[:-4]))
			for i in range(len(filename)):
				name=temp_filepath+filename[i]
				orig=self.mel_CNN(name)
				labels.append(j)
				if switch:
					np.save('C:/Users/peisen/Desktop/cnn_train_mel/'+str(1000*j+i)+'.npy',orig)
				else:
					np.save('C:/Users/peisen/Desktop/cnn_test_mel/'+str(1000*j+i)+'.npy',orig)
		np.save('C:/Users/peisen/Desktop/'+label_name+'.npy',labels)
		print('  OK\n')

	def mel(self,name):
		y,sr=librosa.load(name,sr=None)
		y=self.norm(y)
#		plt.figure()
#		plt.plot([x for x in range(y.shape[0])],y)
#		plt.show()
		zero,ener=self.get_feature(y)
		new_y=self.detect(y,zero,ener,name)
		mfcc_feature=mfcc(signal=new_y,samplerate=sr,winlen=self.len_frame,winstep=(1-self.ratio)*self.len_frame,
						  numcep=self.n_mfcc,nfilt=26,nfft=2000,winfunc=np.hamming)
#		plt.matshow(mfcc_feature)
#		plt.show()
		return mfcc_feature

	def mel_CNN(self,name):
		y,sr=librosa.load(name,sr=None)
		y=self.norm(y)
#		plt.figure()
#		plt.plot([x for x in range(y.shape[0])],y)
#		plt.show()
		zero,ener=self.get_feature(y)
		new_y=self.detect(y,zero,ener,name)
#		plt.figure()
#		plt.plot([x for x in range(new_y.shape[0])],new_y)
#		plt.show()
		new_y=self.padding(new_y)
		mfcc_feature=mfcc(signal=new_y,samplerate=sr,winlen=self.len_frame,winstep=(1-self.ratio)*self.len_frame,
						  numcep=self.n_mfcc,nfilt=26,nfft=2000,winfunc=np.hamming)
#		plt.matshow(mfcc_feature.T)
#		plt.show()
		return mfcc_feature

	def padding(self,y):
		m=len(y)
		if m<self.len_CNN:
			pad_up=int((self.len_CNN-m)/2)
			pad_down=self.len_CNN-pad_up-m
			zero_up=np.zeros((pad_up,))
			zero_down=np.zeros((pad_down,))
			new=np.hstack((zero_up,np.hstack((y,zero_down))))
			return new
		elif m>self.len_CNN:
			y=list(y)
			dec_up=int((m-self.len_CNN)/2)
			dec_down=m-self.len_CNN-dec_up
			for i in range(dec_up):
				y.pop(0)
			for j in range(dec_down):
				y.pop()
			return np.array(y)

	def norm(self,y):
		a=max(max(y),abs(min(y)))
		for i in range(len(y)):
			y[i]=y[i]/a
		return y

	def detect(self,y,zero,ener,name):
		zero_new=self.inter_value(len(y),zero)
		ener_new=self.inter_value(len(y),ener)
		N2,N3,N4,N5=self.ener_detect(len(y),ener_new)
#		N1,N6=self.zero_detect(N2,N5,zero_new)
		new=y[N2:N5]
#		plt.figure()
#		plt.plot([x for x in range(new.shape[0])],new)
#		plt.show()
#		plt.figure()
#		plt.plot([x for x in range(zero_new.shape[0])],zero_new)
#		plt.show()
#		plt.figure()
#		plt.plot([x for x in range(ener_new.shape[0])],ener_new)
#		plt.show()
		return new

	def ener_detect(self,length,ener):
		T2=max(ener)/100
		T1=max(ener)/1000
		temp=0
		while temp >= 0 and temp <= len(ener)-1:
			if ener[temp] < T2:
				temp+=1
			else:
				break
		N3=temp
		temp=length-1
		while temp >= 0 and temp <= len(ener)-1:
			if ener[temp] < T2:
				temp-=1
			else:
				break
		N4=temp
		temp=N3
		while temp > 0:
			if ener[temp] > T1:
				temp-=1
			else:
				break
		N2=temp
		temp=N4
		while temp < len(ener)-1:
			if ener[temp] > T1:
				temp+=1
			else:
				break
		N5=temp
		return N2,N3,N4,N5

	def zero_detect(self,N2,N5,zero):
		T3=max(zero)/3
		temp=N2
		while temp > 0:
			if zero[temp] > T3:
				temp-=1
			else:
				break
		N1=temp
		temp=N5
		while temp < len(zero)-1:
			if zero[temp] > T3:
				temp+=1
			else:
				break
		N6=temp
		return N1,N6

	def inter_value(self,target_len,current_list):
#		plt.figure()
#		plt.plot([x for x in range(len(current_list))],current_list)
#		plt.show()
		x=np.linspace(0,len(current_list)-1,len(current_list))
		x_new=np.linspace(0,len(current_list)-1,target_len)
		f=interpolate.splrep(x,current_list)
		new=interpolate.splev(x_new,f)
		return new

	def get_feature(self,vector):
		zero=[]
		energy=[]
		n=len(vector)
		j=0
		while j*(1-self.ratio)*self.len_points+self.len_points < n:
			start=int(j*(1-self.ratio)*self.len_points)
			end=int(start+self.len_points)
			rate=self.zero_rate(vector[start:end])
			zero.append(rate)
			ener=self.aver_energy(vector[start:end])
			energy.append(ener)
			j+=1
		return zero,energy

	def aver_energy(self,vector):
		energy=0
		for j in range(len(vector)):
			energy+=np.square(vector[j])
		return energy

	def zero_rate(self,vector):
		rate=0
		for j in range(len(vector)-1):
			if vector[j]*vector[j+1]<0:
				rate+=1
		return rate

	def change_name(self):
		filepath='C:/Users/peisen/Desktop/vedio/'
		for i in range(10):
			temp_path=filepath+str(i)+'/'
			filename=os.listdir(temp_path)
			for j in range(len(filename)):
				name=temp_path+filename[j]
				new_name=temp_path+str(j+1)+'.wav'
				os.rename(name,new_name)

	def load_data(self,switch=True,case=True):
		if switch:
			filepath_orig='orig_mel/'
			label_name='train.npy'
			num=self.num_samples
		else:
			filepath_orig='test_mel/'
			label_name='test.npy'
			num=self.test_num
		if case:
			print('\n'+15*'*'+'\n'+'  Loading Data'+'\n'+15*'*')
		filename_orig=os.listdir(filepath_orig)
		filename_orig.sort(key=lambda x:int(x[:-4]))
		features_orig=[]
		for i in range(len(filename_orig)):
			name=filepath_orig+filename_orig[i]
			features_orig.append(np.load(name))
		labels=np.load(label_name)
		if case:
			print('  OK')
		return features_orig,labels

	def dtw_classify(self):
		error=0
		ref,ref_labels=self.load_data()
		test,test_labels=self.load_data(switch=False)
		cost=[]
		aver=[]
		for i in range(len(test)):
			for j in range(len(ref)):
				distance,path=fastdtw(ref[j],test[i],dist=euclidean)
				cost.append(distance)
				if (j+1)%112==0:
					aver.append(sum(cost)/112)
					cost=[]
			pred=np.argmin(aver)
			aver=[]
#			pred=self.knn(10,ref_labels,cost)
			if pred != test_labels[i]:
				error+=1
				print(pred,test_labels[i])
			else:
				print('Right')
		acc=1-error/self.test_num
		print('\nTOTAL ACC:',acc)

	def knn(self,k,labels,distance):
		first=labels[np.argmin(distance)]
		temp=[]
		Inf=999999
		for i in range(k):
			index=np.argmin(distance)
			temp.append(labels[index])
			distance[index]=Inf
		result=Counter(temp).most_common(5)
		print(result)
		if (result[0])[1]>1:
			return int((result[0])[0])
		else:
			return first

	def pre_CNN(self,switch=True):
		if switch:
			filepath_orig='cnn_train_mel/'
			label_name='cnn_train.npy'
			num=self.num_samples
		else:
			filepath_orig='cnn_test_mel/'
			label_name='cnn_test.npy'
			num=self.test_num
		print('\n'+15*'*'+'\n'+'  Loading Data'+'\n'+15*'*')
		filename_orig=os.listdir(filepath_orig)
		filename_orig.sort(key=lambda x:int(x[:-4]))
		features_orig=np.empty(shape=[0,20])
		for i in range(len(filename_orig)):
			name=filepath_orig+filename_orig[i]
			mfccs=np.load(name)
			features_orig=np.append(features_orig,mfccs,axis=0)
		features_orig=features_orig.reshape((num,28,20,1))
		labels=np.load(label_name)
		labels=to_categorical(labels)
		print('  OK')
		return features_orig,labels

	def CNN(self):
		train,trlabels=self.pre_CNN(switch=True)
		test,telabels=self.pre_CNN(switch=False)
		model=models.Sequential()
		model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,20,1)))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(128,(3,3),activation='relu'))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(128,(3,3),activation='relu'))
		model.add(layers.Flatten())
		model.add(layers.Dense(512,activation='relu'))
#		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(128,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(10,activation='softmax'))
		model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
		model.fit(train,trlabels,epochs=20,batch_size=40)
		model.save('cnn5.h5')
		test_loss,test_acc=model.evaluate(test,telabels)
		print('\n')
		print('TEST ACC: ',test_acc)

	def realtime_cnn(self,mfccs):
		m,n=mfccs.shape
		mfccs=mfccs.reshape((1,m,n,1))
		result=[]
		for j in range(self.num_net):
			pred=self.model[j].predict(mfccs)
			if max(pred[0])>0.5:
				predict=np.argmax(pred[0])
				result.append(predict)
		final=Counter(result).most_common()
		if len(final) and (final[0])[1] >= 2:
			print(11*'*'+' ',(final[0])[0],'  '+11*'*'+'\n')
		else:
			print('****  Not Clear && Try Again  ****\n')

	def realtime_dtw(self,mfccs):
		ref,ref_labels=self.load_data(case=False)
		num=int(self.num_samples/10)
		aver=[]
		cost=[]
		for j in range(len(ref)):
			distance,path=fastdtw(ref[j],mfccs,dist=euclidean)
			cost.append(distance)
			if (j+1)%num==0:
				aver.append(sum(cost)/num)
		pred=np.argmin(aver)
		print(pred,'\n')

	def delay(self,k):
		i=0;j=0;l=0
		while i<k:
			while j<k:
				while l<k:
					l+=1
				j+=1
			i+=1

	def real_time(self,fun):
		print('\n'+20*'*'+'  Welcome to speech reconition  '+20*'*'+'\n')
		name='C:/Users/peisen/Desktop/temp/temp.wav'
		string='\nContinue?'+'   [y/n]   '
		st='Start?'+'   [y/n]   '
		switch='n'
		while switch=='n':
			switch=input(st)
			if switch=='n':
				self.delay(9999999)
			else:
				self.delay(1000000)
		if fun=='cnn':
			while switch!='n':
				self.record()
				print(11*'*'+'  Get MFCC  '+11*'*')
				mfccs=self.mel_CNN(name)
				print(11*'*'+'  Finished  '+11*'*'+'\n')
				self.realtime_cnn(mfccs)
				switch=input(string)
		elif fun=='dtw':
			while switch=='y':
				self.record()
				mfccs=self.mel(name)
				self.realtime_dtw(mfccs)
				switch=input(string)
		print('\n'+15*'*'+' Have a good day '+15*'*'+'\n')

if __name__ == '__main__':
	f=FREQ()
#	f.change_name()
#	f.change_freq()
#	f.change_freq(switch=False)
#	f.pre_feature_cnn()
#	f.pre_feature_cnn(switch=False)
#	f.pre_feature_dtw()
#	f.pre_feature_dtw(switch=False)
#	f.CNN()
#	f.dtw_classify()
	f.real_time(fun='cnn')
#	f.real_time(fun='dtw')

