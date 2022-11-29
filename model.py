import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

class D_SAConvNet(object):
    def __init__(self):
        self.name = 'D_SAConvNet'
        
    def __call__(self, inputs, train, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()            
            h1 = tcl.conv2d(inputs, num_outputs=16, kernel_size=5, stride=1, padding='VALID', weights_initializer=tf.random_normal_initializer(0, 0.01),
                            biases_initializer = tf.constant_initializer(0.1), activation_fn=tf.nn.relu)
            p1 = tcl.max_pool2d(inputs=h1, kernel_size=2, stride=2)
            h2 = tcl.conv2d(p1, num_outputs=32, kernel_size=5, stride=1, padding='VALID', weights_initializer=tf.random_normal_initializer(0, 0.01),
                            biases_initializer = tf.constant_initializer(0.1), activation_fn=tf.nn.relu)
            p2 = tcl.max_pool2d(inputs=h2, kernel_size=2, stride=2)
            h3 = tcl.conv2d(p2, num_outputs=64, kernel_size=6, stride=1, padding='VALID', weights_initializer=tf.random_normal_initializer(0, 0.01),
                            biases_initializer = tf.constant_initializer(0.1), activation_fn=tf.nn.relu)
            p3 = tcl.max_pool2d(inputs=h3, kernel_size=2, stride=2)
            h4 = tcl.conv2d(p3, num_outputs=128, kernel_size=5, stride=1, padding='VALID', weights_initializer=tf.random_normal_initializer(0, 0.01),
                            biases_initializer = tf.constant_initializer(0.1), activation_fn=tf.nn.relu)            
            d4 = tcl.dropout(inputs=h4, keep_prob=0.5, is_training=train)
            h5 = tcl.conv2d(d4, num_outputs=11, kernel_size=3, stride=3, padding='VALID', weights_initializer=tf.random_normal_initializer(0, 0.01),
                            biases_initializer = tf.constant_initializer(0), activation_fn=None)
            h5_shape = h5.get_shape().as_list()
            nodes = h5_shape[1]*h5_shape[2]*h5_shape[3]
            outputs = tf.reshape(h5, [-1, nodes])
            
            return outputs
    
    @property        
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class G_SAConvNet(object):
    def __init__(self):
        self.name = 'G_SAConvNet'
        
    def __call__(self, z):
		with tf.variable_scope(self.name) as scope:			
			f = tcl.fully_connected(inputs=z, num_outputs=11*11*128, activation_fn = tf.nn.relu, normalizer_fn=tcl.batch_norm,
									weights_initializer=tf.random_normal_initializer(0, 0.02))
			g0 = tf.reshape(f, (-1, 11, 11, 128)) # 11x11x128
			g1 = tcl.conv2d_transpose(inputs=g0, num_outputs=64, kernel_size=4, stride=2, padding='SAME', # 22x22x64
									  activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(0, 0.02))
			g2 = tcl.conv2d_transpose(inputs=g1, num_outputs=32, kernel_size=4, stride=2, padding='SAME', # 44x44x32
									  activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(0, 0.02))
			g3 = tcl.conv2d_transpose(inputs=g2, num_outputs=1,  kernel_size=4, stride=2, padding='SAME', # 88x88x1
									  activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
			return g3*255
			
    @property        
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)      
