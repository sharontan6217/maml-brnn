
import keras
import tensorflow as tf


meta_step_size = 0.25
eval_interval = 1

def maml_train( model_predict,tasks, meta_optimizer, inner_lr, meta_steps, inner_steps):

    for meta_step in range(meta_steps):
        meta_loss = 0.0
        with tf.GradientTape() as tape:
            for task in tasks:
                # Clone model for task-specific adaptation
                task_model=model_predict
                #task_model.load_state_dict(model_predict.state_dict())
                #task_optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)
                task_optimizer = keras.optimizers.Adam(learning_rate=inner_lr)
                frac_done = meta_step / meta_steps
                cur_meta_step_size = (1 - frac_done) * meta_step_size
                # Temporarily save the weights from the model.
                old_vars = model_predict.get_weights()
                X = task['X']
                Y = task['Y']
                # Inner loop: task-specific adaptation
                for _ in range(inner_steps):
                    with tf.GradientTape() as tape_inner:
                        outputs = model_predict(X)
                        task_loss = keras.losses.mean_squared_error(Y, outputs)
                        print('task loss is: ',task_loss)
                        grads = tape_inner.gradient(task_loss, model_predict.trainable_weights)
                        task_optimizer.apply_gradients(zip(grads, model_predict.trainable_weights))
                        new_vars = model_predict.get_weights()
                        # Perform SGD for the meta step.
                        for var in range(len(new_vars)):
                            new_vars[var] = old_vars[var] + (
                                (new_vars[var] - old_vars[var]) * cur_meta_step_size
                            )
                        # After the meta-learning step, reload the newly-trained weights into the model.
                        model_predict.set_weights(new_vars)

                        # Compute meta-loss
                        meta_loss += task_loss

        # Meta-optimization step
        grads = tape.gradient(meta_loss, model_predict.trainable_weights)
        meta_optimizer.apply_gradients(zip(grads, model_predict.trainable_weights))
        new_vars = model_predict.get_weights()
        # Perform SGD for the meta step.
        for var in range(len(new_vars)):
            new_vars[var] = old_vars[var] + ((new_vars[var] - old_vars[var]) * cur_meta_step_size)
        # After the meta-learning step, reload the newly-trained weights into the model.
        model_predict.set_weights(new_vars)
        meta_step+=1
