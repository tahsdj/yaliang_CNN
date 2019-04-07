from models import NiN
from datahandler import DataHandler, getData

def main():
    xs = tf.placeholder(tf.float32, [None, 200*200*3])   # 28x28
    ys = tf.placeholder(tf.float32, [None, 3])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 200, 200, 3])
    
    # create model
    yaliang_CNN = NiN()
    model_softmax_soft, model_pred, model_attention_map = yaliang_CNN.build(x_image,reuse=None)
    yaliang_CNN.build_loss_fn(ys)
    

    # get data
    training_data, test_data = getData()

    # build dataset 
    dataset = DataHandler(training_data, test_data)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    model_saver = tf.train.Saver()

    # with tf.Session() as sess:
    #     sess.run(init)
    #     #   model_saver.save(sess, './NiN_teacher_init')
        
    #     dataset.reset()
    #     stop_epoch = 100
    #     learning_curve = []
    #     for i in range(10000):
    #         batch_xs, batch_ys = dataset.next_batch(64)
    #         batch_xs = np.reshape(batch_xs,(-1,200*200*3))
    #     #     print(batch_xs.shape)
    #     #     print(batch_ys.shape)

    #         if dataset.epoch >= 0:
    #             lr = 1e-3
    #         if dataset.epoch >= 0.75*stop_epoch:
    #             lr = 1e-4
    #         if dataset.epoch >= 0.9*stop_epoch:
    #             lr = 1e-5
            
    #         feed_dict = {
    #             xs: batch_xs,
    #             ys: batch_ys,
    #             yaliang_CNN.keep_prob: .6,
    #             yaliang_CNN.learning_rate: lr
    #         }
    #         _, t_loss =sess.run([yaliang_CNN.train_step,yaliang_CNN.loss],feed_dict)
            
    #         if i % 50 == 0:
    #             test_xs, test_ys = dataset.get_test_data()
    #             test_xs = np.reshape(test_xs,(-1,200*200*3))
    #             test_acc = yaliang_CNN.compute_accuracy(test_xs, test_ys)
    #             learning_curve.append((test_acc, t_loss))
            
    #         if i % 100 == 0:
    #             test_xs, test_ys = dataset.get_test_data()
    #             test_xs = np.reshape(test_xs,(-1,200*200*3))
    #             test_acc = yaliang_CNN.compute_accuracy(test_xs, test_ys)
            
    #             print('iteration: {:}  epoch: {:}  loss: {:}  acc: {:}'.format(i, dataset.epoch, t_loss, test_acc))
                
    #         if dataset.epoch >= 100:
    #             break
            
            
    #     #   model_saver.save(sess, './NiN_teacher_pretrained')
    #     print('training done~')
    #     print('test acc: {:}'.format(test_acc))
    #     model_saver.save(sess, './models/yaliang_CNN')

if __name__ == '__main__':
    main()