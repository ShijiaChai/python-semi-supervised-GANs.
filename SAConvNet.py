import os
import random
import numpy as np
from glob import glob
import tensorflow as tf
from model import *
import scipy.misc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

INPUT_DATA_PATH = '/home/SemiRecogNet/MSTAR/MSTAR_Ten_Categories_Separate'
MODEL_SAVE_PATH = "model"
RESULT_SAVE_PATH = "result"
SAMPLE_NUM_PER_CLASS = 10
ITERATION_NUM = 40000
CLASS_NUM = 10
DEBUG = False
IM_HEIGHT = 88
IM_WIDTH = 88
IM_CDIM = 1
VAL_RATIO = 0.1
BALANCE_WEIGHT_BASE = 5.0
BALANCE_WEIGHT_DECAY = 0.8
BALANCE_WEIGHT_DECAY_STEP = 1000
D_LEARNING_RATE_BASE = 0.0001
D_LEARNING_RATE_DECAY = 0.99
D_LEARNING_RATE_DECAY_STEP = 5000
G_LEARNING_RATE_BASE = 0.0002
G_LEARNING_RATE_DECAY = 0.99
G_LEARNING_RATE_DECAY_STEP = 5000
G_TRAIN_TIMES_IN_ONE_ITER = 2
COMPUTE_LOSS_INTERVAL = 100
TEST_INTERVAL = 1000
SAVE_MODEL_INTERVAL = 1000
SAVE_GENERATED_DATA_INTERVAL = 1000

flags = tf.app.flags
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def sample_z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])
    
def semi_loss_with_logits(logits, isReturnFakeLoss=True):
    logits_shape = logits.get_shape().as_list()
    logits_softmax = tf.nn.softmax(logits)
    if isReturnFakeLoss:
        loss = tf.log(logits_softmax[:,logits_shape[1]-1]+1e-10)
    else:
        loss = tf.log(1-logits_softmax[:,logits_shape[1]-1]+1e-10)
    if DEBUG:
        print loss
    return loss

def data2fig(samples):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05, hspace=0.05)
        
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(IM_HEIGHT, IM_WIDTH), cmap='Greys_r')
    return fig

class SAConvNet():
    def __init__(self):
        self.category_name = ['2S1', 'BMP2', 'BRDM_2', 'BTR70', 'BTR_60', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU_234']
        self.name_to_idx = {'2S1':0, 'BMP2':1, 'BRDM_2':2, 'BTR70':3, 'BTR_60':4, 'D7':5, 'T62':6, 'T72':7, 'ZIL131':8, 'ZSU_234':9}
        self.train_im_list = {}
        self.train_im_list['labeled'] = {}
        self.train_im_list['unlabeled'] = {}
        self.val_im_list = {}
        self.test_im_list = {}
        for cate_name in self.category_name:
            self.get_category_list(cate_name, 'train_labeled')
            self.get_category_list(cate_name, 'train_unlabeled')
            self.get_category_list(cate_name, 'test')

        if DEBUG:
            print 'The labeled training data number:'
            for cate_name in self.category_name:
                print cate_name, len(self.train_im_list['labeled'][cate_name])
            print 'The unlabeled training data number:'
            for cate_name in self.category_name:
                print cate_name, len(self.train_im_list['unlabeled'][cate_name])
            print 'The validation data number:'
            for cate_name in self.category_name:
                print cate_name, len(self.val_im_list[cate_name])
            print 'The test data number:'
            for cate_name in self.category_name:
                print cate_name, len(self.test_im_list[cate_name])
                print cate_name, len(self.test_im_list[cate_name])
        self.train_list_idx = {}
        self.train_list_idx['labeled'] = {}
        self.train_list_idx['unlabeled'] = {}
        self.train_list_idx["unlabeled"] = self.net
        self.train,list_idx["unlabeled"] = self.ne
        for cate_name in self.category_name:
            self.train_list_idx['labeled'][cate_name] = 0
            self.train_list_idx['unlabeled'][cate_name] = 0        
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        if not os.path.exists(RESULT_SAVE_PATH):
            os.makedirs(RESULT_SAVE_PATH)
        self.loss_output_filename = os.path.join(RESULT_SAVE_PATH, 'loss_output.txt')
        self.test_accuracy_filename = os.path.join(RESULT_SAVE_PATH, 'test_accuracy.txt')
        self.test_im_num = 0
        for cate_name in self.category_name:
            self.test_im_num = self.test_im_num + len(self.test_im_list[cate_name])
        self.val_im_num = 0
        for cate_name in self.category_name:
            self.val_im_num = self.val_im_num + len(self.val_im_list[cate_name])
        # data and label of the labeled data
        self.inputs_labeled = tf.placeholder(tf.float32, shape=[None, IM_HEIGHT, IM_WIDTH, IM_CDIM])
        self.labels = tf.placeholder(tf.float32, shape=[None, CLASS_NUM+1])
        # unlabeled data
        self.inputs_unlabeled = tf.placeholder(tf.float32, shape=[None, IM_HEIGHT, IM_WIDTH, IM_CDIM])
        # input of the generator
        self.z_dim = 100
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        # discriminator and generator of the net
        self.discriminator = D_SAConvNet()
        self.generator = G_SAConvNet()
        # compute the discriminator outputs of labeled data, unlabeled data, and fake data, where fake data is generated from the generator
        self.inputs_fake = self.generator(self.z)
        self.outputs_labeled = self.discriminator(self.inputs_labeled, train=FLAGS.train)
        self.outputs_unlabeled = self.discriminator(self.inputs_unlabeled, train=FLAGS.train, reuse=True)
        self.outputs_fake = self.discriminator(self.inputs_fake, train=FLAGS.train, reuse=True)        
        # learning rate and balance weight        
        self.global_step = tf.Variable(0, trainable=False)
        self.d_learning_rate = tf.train.exponential_decay(D_LEARNING_RATE_BASE, self.global_step, D_LEARNING_RATE_DECAY_STEP, D_LEARNING_RATE_DECAY, staircase = True)
        self.g_learning_rate = tf.train.exponential_decay(G_LEARNING_RATE_BASE, self.global_step, G_LEARNING_RATE_DECAY_STEP, G_LEARNING_RATE_DECAY, staircase = True)
        self.balance_weight = tf.train.exponential_decay(BALANCE_WEIGHT_BASE, self.global_step, BALANCE_WEIGHT_DECAY_STEP, BALANCE_WEIGHT_DECAY, staircase = True)
        # compute the loss
        self.d_loss_labeled = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs_labeled, labels=self.labels))
        self.d_loss_unlabeled = -tf.reduce_mean(semi_loss_with_logits(logits=self.outputs_unlabeled, isReturnFakeLoss=False))
        self.d_loss_fake = -tf.reduce_mean(semi_loss_with_logits(logits=self.outputs_fake, isReturnFakeLoss=True))
        self.d_loss = 0.01*self.d_loss_labeled + self.d_loss_unlabeled + self.d_loss_fake
        self.g_loss = -tf.reduce_mean(semi_loss_with_logits(logits=self.outputs_fake, isReturnFakeLoss=False))
        # compute the accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.outputs_labeled[:,0:CLASS_NUM], 1), tf.argmax(self.labels[:,0:CLASS_NUM], 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))                
        # save all saved model               
        self.saver = tf.train.Saver(max_to_keep=0)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                
    def get_category_list(self, cate_name, stage):
        if stage == 'train_labeled':
            category_list = glob(os.path.join(INPUT_DATA_PATH, 'Degree_17_Train', cate_name, cate_name+'JPGAugL', '*.jpg'))
            random.shuffle(category_list)
            val_num = int(round(len(category_list) * VAL_RATIO))
            self.val_im_list[cate_name] = category_list[0:val_num]
            self.train_im_list['labeled'][cate_name] = category_list[val_num:]
        elif stage == 'train_unlabeled':
            category_list = glob(os.path.join(INPUT_DATA_PATH, 'Degree_17_Train', cate_name, cate_name+'JPGAugU', '*.jpg'))
            random.shuffle(category_list)
            self.train_im_list['unlabeled'][cate_name] = category_list
        else:
            category_list = glob(os.path.join(INPUT_DATA_PATH, 'Degree_15_Test', cate_name, cate_name+'JPG', '*.jpg'))            
            self.test_im_list[cate_name] = category_list
    
    def get_next_batch_list(self, label_type, cate_name):    
        idx = self.train_list_idx[label_type][cate_name]
        num = len(self.train_im_list[label_type][cate_name])
        if idx + SAMPLE_NUM_PER_CLASS > num:
            self.batch_list[label_type][cate_name] = self.train_im_list[label_type][cate_name][idx:num]
            # shuffle the data in one category
            random.shuffle(self.train_im_list[label_type][cate_name])
            new_list = self.train_im_list[label_type][cate_name][0:SAMPLE_NUM_PER_CLASS-(num-idx)]
            self.batch_list[label_type][cate_name] = self.batch_list[label_type][cate_name] + new_list
            self.train_list_idx[label_type][cate_name] = SAMPLE_NUM_PER_CLASS-(num-idx)
        else:
            self.batch_list[label_type][cate_name] = self.train_im_list[label_type][cate_name][idx:idx+SAMPLE_NUM_PER_CLASS]
            self.train_list_idx[label_type][cate_name] = idx+SAMPLE_NUM_PER_CLASS
        if DEBUG:
            print idx    
            
    def train(self):
        d_train_step = tf.train.AdamOptimizer(self.d_learning_rate).minimize(self.d_loss, var_list=self.discriminator.vars, global_step=self.global_step)
        g_train_step = tf.train.AdamOptimizer(self.g_learning_rate).minimize(self.g_loss, var_list=self.generator.vars)
        self.sess.run(tf.global_variables_initializer())
        
        # load the model if there is one
        could_load, checkpoint_counter = self.load()
        if could_load:
            self.global_step.assign(checkpoint_counter)
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        for iter_num in range(ITERATION_NUM+1):
            self.batch_list = {}
            self.batch_list['labeled'] = {}
            self.batch_list['unlabeled'] = {}
            for cate_name in self.category_name:
                self.get_next_batch_list('labeled', cate_name)
                self.get_next_batch_list('unlabeled', cate_name)
            batch_filename = {}
            batch_filename['labeled'] = []
            batch_filename['unlabeled'] = []
            batch_cls = []
            for cate_name in self.category_name:
                batch_filename['labeled'] = batch_filename['labeled'] + self.batch_list['labeled'][cate_name]
                cls_ind = self.name_to_idx[cate_name]
                batch_cls = batch_cls + [cls_ind]*SAMPLE_NUM_PER_CLASS
                batch_filename['unlabeled'] = batch_filename['unlabeled'] + self.batch_list['unlabeled'][cate_name]
            if DEBUG:
                print batch_filename['labeled']
                print batch_filename['unlabeled']
                print batch_cls
            # shuffle the sample in one batch
            temp = list(zip(batch_filename['labeled'], batch_cls))
            random.shuffle(temp)
            batch_filename['labeled'][:], batch_cls[:] = zip(*temp)
            random.shuffle(batch_filename['unlabeled'])
            if DEBUG:
                print batch_cls
            inputs_labeled = np.zeros((len(batch_filename['labeled']), IM_HEIGHT, IM_WIDTH, IM_CDIM))
            inputs_labeled = inputs_labeled.astype('float32')
            labels = np.zeros((len(batch_filename['labeled']), CLASS_NUM+1))
            labels = labels.astype('float32')
            inputs_unlabeled = np.zeros((len(batch_filename['unlabeled']), IM_HEIGHT, IM_WIDTH, IM_CDIM))
            inputs_unlabeled = inputs_unlabeled.astype('float32')
            if DEBUG:
                print inputs_labeled.shape
                print inputs_unlabeled.shape
                print labels.shape
                            
            for k in range(len(batch_filename['labeled'])):
                im = scipy.misc.imread(batch_filename['labeled'][k])
                im = im.reshape(IM_HEIGHT, IM_WIDTH, IM_CDIM).astype('float32')
                if DEBUG:
                    print im.shape
                    print im.dtype
                inputs_labeled[k] = im
                labels[k][batch_cls[k]] = 1
            for k in range(len(batch_filename['unlabeled'])):
                im = scipy.misc.imread(batch_filename['unlabeled'][k])
                im = im.reshape(IM_HEIGHT, IM_WIDTH, IM_CDIM).astype('float32')
                if DEBUG:
                    print im.shape
                    print im.dtype
                inputs_unlabeled[k] = im
            z = sample_z(len(batch_filename['labeled']), self.z_dim)
            
            if DEBUG:
                output = self.sess.run(self.output, feed_dict={self.inputs_labeled:inputs_labeled, self.labels:labels})
                print output.shape

            counter = self.sess.run(self.global_step)
            balance_weight = self.sess.run(self.balance_weight)    
            self.sess.run(d_train_step, feed_dict={self.inputs_labeled:inputs_labeled, self.labels:labels, 
                                                   self.inputs_unlabeled:inputs_unlabeled, self.z:z})
            for _ in range(G_TRAIN_TIMES_IN_ONE_ITER):
                self.sess.run(g_train_step, feed_dict={self.z:z})
            
            # Display losses and restore them every COMPUTE_LOSS_INTERVAL iterations
            if counter % COMPUTE_LOSS_INTERVAL == 0:
                d_loss_labeled, d_loss_unlabeled, d_loss_fake, d_rate, g_rate = self.sess.run([self.d_loss_labeled, self.d_loss_unlabeled, self.d_loss_fake, 
                                                                                               self.d_learning_rate, self.g_learning_rate], 
                                                                                              feed_dict={self.inputs_labeled:inputs_labeled, self.labels:labels, 
                                                                                                         self.inputs_unlabeled:inputs_unlabeled, self.z:z})
                g_loss = self.sess.run(self.g_loss, feed_dict={self.z:z})
                print("counter:[%2d], d_loss_labeled:%.8f, d_loss_unlabeled:%.8f, d_loss_fake:%.8f, g_loss:%.8f, d_rate:%.8f, g_rate:%.8f, balance_weight:%.8f") \
                      % (counter, d_loss_labeled, d_loss_unlabeled, d_loss_fake, g_loss, d_rate, g_rate, balance_weight)
            
                if DEBUG:
                    outputs_labeled = self.sess.run(self.outputs_labeled, feed_dict={self.inputs_labeled:inputs_labeled, self.labels:labels})
                    for k in range(len(outputs_labeled)):
                        print k
                        print outputs_labeled[k]
                        print labels[k]
                        
                if counter == 0:
                    fid = open(self.loss_output_filename, 'w')
                else:
                    fid = open(self.loss_output_filename, 'a')
                fid.write('{} {} {} {} {} {} {}\n'.format(counter, d_loss_labeled, d_loss_unlabeled, d_loss_fake, g_loss, d_rate, g_rate))
                fid.close()
                
            # Do one test every TEST_INTERVAL iterations, output accuracies of the validation set and the test set respectively   
            if counter % TEST_INTERVAL == 0:
                # The validation set
                val_inputs = np.zeros((self.val_im_num, IM_HEIGHT, IM_WIDTH, IM_CDIM))
                val_inputs = val_inputs.astype('float32')
                val_labels = np.zeros((self.val_im_num, CLASS_NUM+1))
                val_labels = val_labels.astype('float32')
                nn = 0
                for cate_name in self.category_name:
                    for kk in range(len(self.val_im_list[cate_name])):
                        im = scipy.misc.imread(self.val_im_list[cate_name][kk])
                        im = im.reshape(IM_HEIGHT, IM_WIDTH, IM_CDIM).astype('float32')
                        val_inputs[nn] = im
                        val_labels[nn][self.name_to_idx[cate_name]] = 1
                        nn = nn + 1                
                val_accuracy = self.sess.run(self.accuracy, feed_dict={self.inputs_labeled:val_inputs, self.labels:val_labels})
                print("The validation set accuracy is {}".format(val_accuracy))
                # The test set                
                test_inputs = np.zeros((self.test_im_num, IM_HEIGHT, IM_WIDTH, IM_CDIM))
                test_inputs = test_inputs.astype('float32')
                test_labels = np.zeros((self.test_im_num, CLASS_NUM+1))
                test_labels = test_labels.astype('float32')
                nn = 0
                for cate_name in self.category_name:
                    for kk in range(len(self.test_im_list[cate_name])):
                        im = scipy.misc.imread(self.test_im_list[cate_name][kk])
                        im = im.reshape(IM_HEIGHT, IM_WIDTH, IM_CDIM).astype('float32')
                        test_inputs[nn] = im
                        test_labels[nn][self.name_to_idx[cate_name]] = 1
                        nn = nn + 1                
                test_accuracy = self.sess.run(self.accuracy, feed_dict={self.inputs_labeled:test_inputs, self.labels:test_labels})
                print("The test set accuracy is {}".format(test_accuracy))
                
                if counter == 0:
                    fid = open(self.test_accuracy_filename, 'w')
                else:
                    fid = open(self.test_accuracy_filename, 'a')
                fid.write('{} {} {}\n'.format(counter, val_accuracy, test_accuracy))
                fid.close()
            
            # Save the model every 1000 iterations
            if counter % SAVE_MODEL_INTERVAL == 0:
                self.save(counter)
                
            # Save the generated data every 1000 iterations
            if counter % SAVE_GENERATED_DATA_INTERVAL == 0:
                samples = self.sess.run(self.inputs_fake, feed_dict={self.z:sample_z(16, self.z_dim)})
                fig = data2fig(samples)
                plt.savefig('{}/{}.png'.format(RESULT_SAVE_PATH, str(counter)), bbox_inches='tight')            
                plt.close(fig)

    def test(self):
        # load the trained model
        could_load, checkpoint_counter = self.load()
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        test_im_num = 0
        for cate_name in self.category_name:
            test_im_num = test_im_num + len(self.test_im_list[cate_name])
        inputs = np.zeros((test_im_num, IM_HEIGHT, IM_WIDTH, IM_CDIM))
        inputs = inputs.astype('float32')
        labels = np.zeros((test_im_num, CLASS_NUM+1))
        labels = labels.astype('float32')
        n = 0
        for cate_name in self.category_name:
            for k in range(len(self.test_im_list[cate_name])):
                im = scipy.misc.imread(self.test_im_list[cate_name][k])
                im = im.reshape(IM_HEIGHT, IM_WIDTH, IM_CDIM).astype('float32')
                inputs[n] = im
                labels[n][self.name_to_idx[cate_name]] = 1
                n = n + 1
                
                
        accuracy = self.sess.run(self.accuracy, feed_dict={self.inputs_labeled:inputs, self.labels:labels})
        print("Test accuracy is {}".format(accuracy))
        if True:
            outputs_label = self.sess.run(self.outputs_labeled, feed_dict={self.inputs_labeled:inputs})        
            outputs_label = outputs_label[:,0:CLASS_NUM]
            real_label = labels[:,0:CLASS_NUM]
            confusion_matrix = np.zeros((CLASS_NUM, CLASS_NUM))
            for i in range(len(real_label)):
                row_idx = np.argmax(real_label[i])
                col_idx = np.argmax(outputs_label[i])
                confusion_matrix[row_idx][col_idx] = confusion_matrix[row_idx][col_idx] + 1
            print confusion_matrix
            for i in range(CLASS_NUM):
                correct_num = confusion_matrix[i][i]
                total_num = np.sum(confusion_matrix[i])
                correct_num = correct_num.astype('float32')
                total_num = total_num.astype('float32')
                acc = correct_num/total_num
                print("{}: accuracy is {}/{}={}".format(self.category_name[i], correct_num, total_num, acc))
                        
        if DEBUG:
            samples = self.sess.run(self.inputs_fake, feed_dict={self.z:sample_z(10, self.z_dim)})
            samples = samples.reshape(10, IM_HEIGHT, IM_WIDTH)
            for k in range(10):            
                scipy.misc.imsave('{}/Final_Generate_{}.tiff'.format(RESULT_SAVE_PATH, k), samples[k])
    
    def save(self, step):
        model_name = "SAConvNet.model"
        
        self.saver.save(self.sess, os.path.join(MODEL_SAVE_PATH, model_name), global_step=step)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(MODEL_SAVE_PATH, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(os.path.join(MODEL_SAVE_PATH, ckpt_name)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0                                  

if __name__ == '__main__':
    
    net = SAConvNet()    
    if FLAGS.train:        
        net.train()
    else:
        net.test()

