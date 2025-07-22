import os

#os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import gc
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

meta_step_size = 0.25
eval_interval = 1

def reptile_train(model_predict, tasks, meta_optimizer, meta_steps):
    #gc.collect()
    training = []
    testing = []
    for meta_step in range(meta_steps):
        frac_done = meta_step / meta_steps
        cur_meta_step_size = (1 - frac_done) * meta_step_size
        # Temporarily save the weights from the model.
        old_vars = model_predict.get_weights()
        #train_dataset = tasks[0]
        #for X, Y in tasks[0]:
        X = tasks[0]['X']
        Y = tasks[0]['Y']
        with tf.GradientTape() as tape:
            preds = model_predict(X)
            loss = keras.losses.mean_squared_error(Y, preds)
        grads = tape.gradient(loss, model_predict.trainable_weights)
        meta_optimizer.apply_gradients(zip(grads, model_predict.trainable_weights))
        new_vars = model_predict.get_weights()
        # Perform SGD for the meta step.
        for var in range(len(new_vars)):
            new_vars[var] = old_vars[var] + (
                (new_vars[var] - old_vars[var]) * cur_meta_step_size
            )
        # After the meta-learning step, reload the newly-trained weights into the model.
        model_predict.set_weights(new_vars)
        # Evaluation loop
        if meta_step % eval_interval == 0:
            accuracies = []
            train_set = tasks[0]
            test_set = tasks[1]
            #for X_,Y_ in test_set:
            x_test = test_set['X']
            y_test = test_set['Y']
            old_vars = model_predict.get_weights()
            # Train on the samples and get the resulting accuracies.
            x_train = train_set['X']
            y_train = train_set['Y']
            with tf.GradientTape() as tape:
                preds = model_predict(x_train)
                loss = keras.losses.mean_squared_error(y_train, preds)
                grads = tape.gradient(loss, model_predict.trainable_weights)
                meta_optimizer.apply_gradients(zip(grads, model_predict.trainable_weights))
            diff_train = sum((preds-y_train))
            training.append(diff_train)
            test_preds = model_predict.predict(x_test)
            #test_preds = tf.argmax(test_preds).numpy()
            diff_test = sum((test_preds-y_test))
            # Reset the weights after getting the evaluation accuracies.
            model_predict.set_weights(old_vars)
            testing.append(diff_test)
            if meta_step % 100 == 0:
                print(
                    "batch %d: train=%f test=%f" % (meta_step, diff_train, diff_test)
                )
    return training,testing,model_predict
def reptile_visualize(training,testing,timeSequence,start,graph_dir):
    # First, some preprocessing to smooth the training and testing arrays for display.
    graph_dir = graph_dir+'acuracy/'
    if os.path.exists(graph_dir)==False:
        os.makedirs(graph_dir)
    window_length = 100
    train_s = np.r_[
        training[window_length - 1 : 0 : -1],
        training,
        training[-1:-window_length:-1],
    ][100:]
    test_s = np.r_[
        testing[window_length - 1 : 0 : -1], testing, testing[-1:-window_length:-1]
    ][100:]
    #w = np.hamming(window_length)
    #train_y = np.convolve(w / w.sum(), train_s, mode="valid")
    #test_y = np.convolve(w / w.sum(), test_s, mode="valid")

    # Display the training accuracies.
    x = np.arange(0, len(test_s), 1)
    plt.plot(x, test_s, x, train_s)
    plt.legend(["test", "train"])
    plt.grid()
    plt.title('Plot Graph of Accuracies')
    fig_name='test_scenario_reptile_brnn_'+timeSequence+'_'+str(start)+'.png'
    plt.savefig(graph_dir+fig_name)
    plt.close()
