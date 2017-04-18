import tensorflow as tf
import numpy as np
from glob import glob
import scipy.misc
import cv2
import sys

learning_rate =0.001
training_iters = 200000
batch_size =12
display_step=10

n_epochs=100
n_classes = 18 
dropout = 0.75 # Dropout, probability to keep units

def conv_bias(inp,wshape):
	w = tf.Variable(tf.truncated_normal(shape=wshape,stddev= 0.001))
	b = tf.Variable(tf.constant(0.1,shape=[wshape[3]]))
	conv_op = tf.nn.conv2d(inp,w,strides=[1,1,1,1],padding='SAME')
	conv_op = tf.nn.bias_add(conv_op,b)
	return tf.nn.relu(conv_op)
	
def deconv(inp, wshape, outputshape, stride) :
	w = tf.Variable(tf.truncated_normal(shape=wshape,stddev= 0.1))
	deconv_op = tf.nn.conv2d_transpose(inp,w,outputshape,strides=[1,stride,stride,1],padding='SAME')
	return deconv_op
	
def max_pool(inp, ksize,stridesize):
	pooled_op = tf.nn.max_pool(inp,ksize=[1,ksize,ksize,1], strides =[1,stridesize,stridesize,1],padding ='SAME')
	return pooled_op

def model(image, keep_prob):
	
	conv_1_1 = conv_bias(image,[5,5,3,128])
	
	conv_1_2 = conv_bias(conv_1_1,[5,5,128,191])
	pool_1 = max_pool(conv_1_2,3,2)
	
	conv_2_1 = conv_bias(pool_1,[5,5,191,191])
	conv_2_2 = conv_bias(conv_2_1,[5,5,191,191])
	pool_2 = max_pool(conv_2_2,3,2)
	
	conv_3_1 = conv_bias(pool_2,[5,5,191,191])
	conv_3_2 = conv_bias(conv_3_1,[5,5,191,191])
	pool_3 = max_pool(conv_3_2,3,2)
	
	conv_4_1 = conv_bias(pool_3,[5,5,191,191])
	conv_4_2 = conv_bias(conv_4_1,[5,5,191,191])
	
	conv_5_1 = conv_bias(conv_4_2,[1,1,191,96])
	
	#fully connected and probabilities
	
	fc1wts = tf.Variable(tf.truncated_normal(shape=[13*19*96,1024]))
	fc1bias = tf.Variable(tf.truncated_normal(shape=[1024]))
	fc_1 = tf.add(tf.matmul(tf.reshape(conv_5_1, [-1,fc1wts.get_shape().as_list()[0]]),fc1wts),fc1bias)
	fc_1 = tf.nn.dropout(fc_1,0.70)
	fc2wts = tf.Variable(tf.truncated_normal(shape=[1024,19]))
	fc2bias = tf.Variable(tf.truncated_normal(shape=[19]))
	fc_2 = tf.add(tf.matmul(fc_1,fc2wts),fc2bias)
	softmax_1 = tf.nn.softmax(fc_2)
	softmax_1 = tf.expand_dims(softmax_1, 1)
	softmax_1 = tf.expand_dims(softmax_1, 2)
	
	#deconv1 = deconv(conv_4_2, [2,2,191,191], [1,38,25,191], 2 ) #upsampling
	upsample1 = tf.image.resize_images(conv_5_1,[38,25])
	deconv1 = conv_bias(upsample1,[5,5,96,191])	
	gb_1 = tf.add(deconv1, conv_3_2) #elementwise sum with old activations	
	
	prob_1 = tf.zeros([1, 38,25,19])
	prob_1 = tf.add(prob_1, softmax_1) #probbaility map to be concatinated
	
	gb_concat_1 = tf.concat([gb_1, prob_1],3) #concatinated probability map to summed up activations
	global_1 = conv_bias(gb_concat_1,[5,5,210,191])


	#deconv2 = deconv(global_1, [2,2,191,191], [1,75,50,191], 2 )
	upsample2 = tf.image.resize_images(global_1,[75,50])
	deconv2 = conv_bias(upsample2,[3,3,191,191])
	gb_2 = tf.add(deconv2, conv_2_2)
	prob_2 = tf.zeros([1,75,50,19])
	prob_2 = tf.add(prob_2,softmax_1)
	gb_concat_2 = tf.concat([gb_2, prob_2],3)
	global_2 = conv_bias(gb_concat_2,[5,5,210,191])
	
	#deconv3 = deconv(global_2, [2,2,191,191],[1,150,100,191],2)
	upsample3 = tf.image.resize_images(global_2,[150,100])
	deconv3 = conv_bias(upsample3,[5,5,191,191])
	gb_3 = tf.add(deconv3,conv_1_2)
	prob_3 = tf.zeros([1,150,100,19])
	prob_3 = tf.add(prob_3,softmax_1)
	gb_concat_3 = tf.concat([gb_3,prob_3],3)
	global_3 = conv_bias(gb_concat_3,[5,5,210,191])
	
	image_conv = conv_bias(image, [5,5,3,191])
	global_4 = tf.add(global_3, image_conv)
	global_conv_4 = conv_bias(global_4, [3,3,191,256])
	
	
	pred_conv_1 = conv_bias(global_conv_4, [1,1,256,19])
	pred_sum = tf.add(pred_conv_1, prob_3)
	pred_conv_2 = conv_bias(pred_sum, [1,1,19,18])
	annotation_pred = tf.argmax(pred_conv_2, dimension=3, name="prediction")
		
	return pred_conv_2, annotation_pred
	
	
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    
    return optimizer.apply_gradients(grads)
    
def load_image(image_path):
	img_A = scipy.misc.imread(image_path)
	img_A = scipy.misc.imresize(img_A, [150, 100])
	if(len(img_A.shape)<3):
		img_A = cv2.cvtColor(img_A,cv2.COLOR_GRAY2RGB)
		
	return img_A
def load_data(image_path):
   
	img_A = scipy.misc.imread(image_path)
	imagename = image_path.split('/')[-1]
	img_B = scipy.misc.imread('./Data/SegmentationClassAug/'+imagename.split('.')[0]+'.png')
	img_A = scipy.misc.imresize(img_A, [150, 100])
	img_B = scipy.misc.imresize(img_B, [150, 100])
	if(len(img_A.shape)<3):
		img_A = cv2.cvtColor(img_A,cv2.COLOR_GRAY2RGB)
	#cv2.imshow('image',img_A)
	#cv2.waitKey(0)
	#print(img_A.shape, img_B.shape)
	return img_A, img_B
	
def main(_):
	
	
	keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
	image = tf.placeholder(tf.float32, shape=[None, 150, 100, 3], name="input_image")
	annotation = tf.placeholder(tf.int32, shape=[None, 150, 100, 1], name="annotation")
	predictions, softmax_1=model(image, keep_probability)
    	
	
	tf.summary.image("input_image", image, max_outputs=2)
	tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
	tf.summary.image("pred_annotation", tf.cast(predictions, tf.uint8), max_outputs=2)
	loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
	tf.summary.scalar("entropy", loss)

	trainable_var = tf.trainable_variables()
    
	train_op = train(loss, trainable_var)

	print("Setting up summary op...")
	summary_op = tf.summary.merge_all()
	
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state('./logs')
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("Model restored...")
	
	if(len(sys.argv)>1):
		
		batch_files = [sys.argv[1]]#test_image_files[0:1]
		tr_images= [load_image(batch_file) for batch_file in batch_files]
		tr_images= np.array(tr_images).astype(np.float32)
		feed_dict = {image: tr_images,  keep_probability: 1.00}
		op = sess.run(softmax_1, feed_dict=feed_dict)
		op = np.squeeze(op)
		cv2.imwrite('test.png',op)
		
	else:
		train_image_files = glob('./Data/JPEGImages/*.jpg')
		print(len(train_image_files))
		batch_idxs = len(train_image_files)/batch_size
		cnt=0
		for epoch in range(n_epochs):
			for idx in xrange(0, batch_idxs):
				batch_files = train_image_files[idx*batch_size:(idx+1)*batch_size]
				total_images= [load_data(batch_file) for batch_file in batch_files]
				tr_images = [t[0] for t in total_images]
				tr_annot = [t[1] for t in total_images]
				#print(len(tr_images))
				tr_images= np.array(tr_images).astype(np.float32)
				tr_annot = np.array(tr_annot).astype(np.float32)[:, :, :, None]
				#print(tr_annot.shape)
				feed_dict = {image: tr_images, annotation: tr_annot, keep_probability: 0.85}
				
				sess.run(train_op, feed_dict=feed_dict)
				
				if cnt % 10 == 0:
					train_loss = sess.run(loss, feed_dict=feed_dict)
					print("Step: %d, Train_loss:%g" % (cnt, train_loss))
				
				if cnt % 1000 == 0:
					train_loss = sess.run(loss, feed_dict=feed_dict)
					#print("Step: %d, Train_loss:%g" % (cnt, train_loss))
					saver.save(sess, 'logs/' +str(epoch), cnt)
					#summary_writer.add_summary(summary_str, idx)
					#break
				cnt +=1
	

		
		
		
		

if __name__ == "__main__" :
	tf.app.run()
