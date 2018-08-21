import pickle
import numpy as np
from scipy import ndimage
import os
from PIL import Image
import random


def img2arr(subfolder,start_image=None,end_image=None,image_size=(24,24,3),pixel_depth=255.0):

    '''subfolder = path to subfolder,
       start_image and end_image parameters are to input the range of images
       for example start_image=1,end_image=200 tells the method to
       convert images ranging 1-200 into np arrays of shape (28,28).
    '''

    #list of absolute path of images excluding hidden files
    image_list=[os.path.join(subfolder,file) for file in os.listdir(subfolder)
                              if not file.startswith('.')][start_image:end_image]
    num_images=len(image_list)
    dataset=[]

    for image_name in image_list:
        try:
            #converting each image into np arrays adjusted to mean=0, std=0.5-1
            image_data=(ndimage.imread(image_name).astype(np.float32)-
                    pixel_depth/2)/pixel_depth

            if image_data.shape!=(image_size):
                #excluding useless images
                print('Unexpected image shape: %s' % str(image_data.shape))
                continue
            dataset.append(image_data)

        except IOError as e:
            #excluding useless images
            print('Could not read:',image_name, ':', e, '- it\'s ok, skipping.')
    #return the list of np arrays of images
    return dataset

def one_hot(element,list_of_elements):
    ''' ex:- one_hot('C',['A','B','C','D']) returns [0,0,1,0]
        in your case,
        element = absolute path of a subfolder
        list_of_elements = list of folders in main folder i.e os.listdir(main_folder)
    '''
    k=[0 for i in range(len(list_of_elements))]
    index=list_of_elements.index(element)
    k[index]=1
    return k

def from_main_folder(path,start_image=None,end_image=None,image_size=24):

    '''
    this function is used to iterate through all the subfolders in the mainfolder and
    img2arr method is used to convert images in each subfolder to np.arrays

    path = absolute path to main folder
    start_image,end_image are passed as parameters ti im2arr used inside this function

    i.e start_image=0,end_image=1000 converts the first 1000 images in each subfolder into numpy arrays

    '''

    folder_list=[os.path.join(path,folder) for folder in
                 os.listdir(path) if not folder.startswith('.')]
    labels=[]
    dataset=[]

    for folder in folder_list:
        data_in_branch=img2arr(folder,start_image,end_image)
        dataset=dataset+data_in_branch
        labels=labels+[one_hot(folder,folder_list) for i in range(len(data_in_branch))]

    for i in range(len(dataset)):
        #image arrays are reshaped from (28,28) to (784,)
        #if your dimensions are other than (28,28), lets say (d,d) the reshape it to (d*d)
        dataset[i]=dataset[i].reshape(1728)

    #return a tuple of dataset and labels
    return dataset,labels

def shuffle(dataset,labels):
    #dataset and corresponding labels extracted from from_main_folder method
    # and fed to shuffle data
    to_shuffle=list(zip(dataset,labels))
    random.shuffle(to_shuffle)
    dataset,labels=zip(*to_shuffle)
    return np.array(dataset),np.array(labels).astype(np.float32)

def save(path,data):
	with open('data.pickle', 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print('pickling...')
    # f=open('','wb')
    # pickle.dump(kwargs,f)
    # print('done')

def resize(path):
	dirs = os.listdir( path )
	for item in dirs:
		for i in os.listdir(path+item):
			if os.path.isfile(path+item+'/'+i):
				im = Image.open(path+item+'/'+i).convert('RGB')
				f, e = os.path.splitext(path+item+'/'+i)
				head, tail = os.path.split(path+item+'/'+i)
				imResize = im.resize((24,24), Image.ANTIALIAS)
				imResize.save(head+'/resized/'+tail, 'JPEG', quality=90)


#the implementation

#preprocess user supplied image before inputting in the NN
def preprocess_image(img_path, num_images):
	data,_=from_main_folder('testNN_resized/', 0, int(num_images))
	return data	

def get_data_sets_and_labels():
	path='images/'

	print('working on training set....')
	data,labels=from_main_folder(path,0,50)
	training_dataset,training_labels=shuffle(data,labels)
	# 30000 pics in each subfolder are picked, converted to arrays and shuffled
	print('Done ! \n\n')

	print('working on testing set....')
	data,labels=from_main_folder(path,50,70)
	testing_dataset,testing_labels=shuffle(data,labels)
	''' 30000 to 50000th pic (total 20000 pics) in each subfolder are picked,
	converted to arrays and shuffled '''

	print('Done !')

	dataset={
	'training_dataset':training_dataset,
	'training_labels':training_labels ,
	'testing_dataset':testing_dataset,
	'testing_labels':testing_labels
	}
	#save the dictionary
	save(path, dataset)

	return training_dataset, training_labels, testing_dataset, testing_labels

if __name__=='__main__':
	get_data_sets_and_labels()
    