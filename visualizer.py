import chainer
import matplotlib.pyplot as plt
import numpy as np
import math
import chainer.functions as F
from chainer.links import caffe
from matplotlib.ticker import *
from chainer import cuda

float32=0
i = 1

def plot(layer):
	dim = eval('('+layer.W.label+')')[0]
	print('dim', dim)
	size = int(math.ceil(math.sqrt(dim[0])))
	print('size', size)
	if(len(dim)==4):
		for i,channel in enumerate(layer.W.data):
			print('i', i)
			ax = plt.subplot(size,size, i+1)
			print('ax finish')
			ax.xaxis.set_major_locator(NullLocator())
			ax.yaxis.set_major_locator(NullLocator())
			accum = channel[0]
			for ch in channel:
				accum += ch
			accum /= len(channel)
			print('accum', accum)
			accum = cuda.to_cpu(accum)
			ax.imshow(accum, interpolation='nearest')
	else:
		plt.imshow(layer.W.data, interpolation='nearest')


def showPlot(layer):
	fig = plt.figure()
	fig.patch.set_facecolor('black')
	#fig.suptitle(layer.W.label, fontweight="bold",color="white")
	plot(layer)
	plt.show()

	print('finish plot')

def plot2(layer):
	dim = eval('('+layer.W.label+')')[0]
	print('dim', dim)
	size = int(math.ceil(math.sqrt(dim[0])))
	print('size', size)
	if(len(dim)==4):
		for i,channel in enumerate(layer.W.data):
			print('i', i)
			ax = plt.subplot(size,size, i+1)
			print('ax finish')
			ax.xaxis.set_major_locator(NullLocator())
			ax.yaxis.set_major_locator(NullLocator())
			accum = channel[0]
			for ch in channel:
				accum += ch
			accum /= len(channel)
			print('accum', accum)
			accum = cuda.to_cpu(accum)
			ax.imshow(accum, interpolation='nearest')
	else:
		plt.imshow(layer.W.data, interpolation='nearest')


def plot3(frames):
	for i, frame in enumerate(frames):
		rootDim = math.sqrt(len(frame))
		print 'frame',frame
		frame2d = np.asarray([frame], dtype=np.float32)
		array2d = np.reshape(frame2d, (rootDim,rootDim))
		frameCount = math.ceil(math.sqrt(len(frames)))
		ax = plt.subplot(frameCount, frameCount, i+1)
		ax.yaxis.set_major_locator(NullLocator())
		ax.yaxis.set_major_locator(NullLocator())

		ax.imshow(array2d, interpolation="nearest")



def showPlot2(layer):
	fig = plt.figure()
	fig.patch.set_facecolor('black')
	#fig.suptitle(layer.W.label, fontweight="bold",color="white")
	plot2(layer)
	plt.show()
	plt.savefig('/images/'+str(i))

def showPlot3(frames, name):
	fig = plt.figure()
	fig.patch.set_facecolor('black')
	plot3(frames)
	plt.draw()
	plt.savefig('./features/'+name)

def savePlot(layer,name):
	fig = plt.figure()
	fig.suptitle(name+" "+layer.W.label, fontweight="bold")
	plot(layer)
	plt.draw()
	plt.savefig("./images/"+name+".png")

def savePlot2(layer,name):
	fig = plt.figure()
	fig.patch.set_facecolor('black')
	#fig.suptitle(layer.W.label, fontweight="bold",color="white")
	plot2(layer)
	plt.draw()
	plt.savefig("./images/"+name+".png")

def save(func):
	for candidate in func.layers:
		if(candidate[0]) in dir(func):
			name=candidate[0]
			savePlot(func[name],name)
