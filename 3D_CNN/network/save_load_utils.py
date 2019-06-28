import tensorflow as tf
import os

def save(sess, saver, weights_dir, model_name, step):

    print(" [*] Saving model in path: %s" % weights_dir)
    
    saver.save(sess, os.path.join(weights_dir, model_name + ".model"), global_step = step)
    
    print("Model saved in path: %s" % weights_dir)

def load(sess, saver, weights_dir, model_name):

    print(" [*] Reading checkpoints...")
    
    ckpt = tf.train.get_checkpoint_state(weights_dir)
    
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(weights_dir, ckpt_name))
        counter = int(ckpt_name.split('-')[-1])
        print(" [*] Success to read {}".format(ckpt_name))
    else:
        print(" [*] Failed to find a checkpoint")
        

if __name__ == '__main__':

    import os
    import tensorflow as tf
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
    
    
    #Prepare to feed input, i.e. feed_dict and placeholders
    w1 = tf.placeholder("float", name="w1")
    w2 = tf.placeholder("float", name="w2")
    b1= tf.Variable(2.0,name="bias")
    feed_dict ={w1:4,w2:8}

    #Define a test operation that we will restore
    w3 = tf.add(w1,w2)
    w4 = tf.multiply(w3,b1,name="op_to_restore")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #Create a saver object which will save all the variables
    saver = tf.train.Saver()

    #Run the operation by feeding input
    print(sess.run(w4,feed_dict))
#Prints 24 which is sum of (w1+w2)*b1         

    save(sess, saver, ".", "3D_CNN", 10)
    load(sess, saver, ".", "3D_CNN")