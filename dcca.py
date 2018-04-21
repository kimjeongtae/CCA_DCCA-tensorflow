class DCCA(object):
    def __init__(self, sess, input_dim, hiddens_dim, output_dim, cca_dim, reg, activation):
        self.sess = sess
        self.input_dim = input_dim
        self.hiddens_dim = hiddens_dim
        self.output_dim = output_dim
        self.cca_dim = cca_dim
        self.reg = reg
        self.activation = activation
        self.training = tf.placeholder(tf.bool)   
        self.f_input, self.f_output = self.build_network('f')
        self.g_input, self.g_output = self.build_network('g')
        
        self.U = tf.get_variable('U', [output_dim, cca_dim], trainable=False)
        self.V = tf.get_variable('V', [output_dim, cca_dim], trainable=False)
        
        self.f_cca = tf.matmul(sef.f_output, self.f_proj, name='f_cca')
        self.g_cca = tf.matmul(sef.f_output, self.g_proj, name='g_cca')

    def _build_network(self, name):
        print(f'Building {name} network...')
        input_layer = tf.placeholder(tf.float32, shape=[None, input_dim], name=f'{name}_input')
        for i, hidden_dim in enumerate(self.hiddens_dim, 1):
            if i == 1:
                hidden_layer = tf.layers.dropout(tf.layers.dense(input_layer, hidden_dim, activation=activation), 0.7, name=f'{name}_hidden{i}')
            else:
                hidden_layer = tf.layers.dropout(tf.layers.dense(hidden_layer, hidden_dim, activation=activation), 0.7, name=f'{name}_hidden{i}')
        ouput_layer =  tf.layers.dense(hidden_layer, self.output_dim, name=f'{name}_output')
        return input_layer, output_layer
    
    def _get_batch(x1, x2, batch_size):
        for batch_i in range(len(x1) // batch_size):
            start_i = batch_i * batch_size
            end_i = (batch_i + 1) * batch_size
            yield x1[start_i:end_i], x2[start_i: end_i]
                     
    def train(self, train_data, valid_data=None, learning_rate=0.001, batch_size=128, epochs=100,
              optimizer=tf.train.AdamOptimizer, save_path='', load_path='', display_size=5, save_size=5):
        
        if load_path:
            self.load(load_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        train_loss_history = []
        valid_loss_history = []
        
        train_x1, train_x2 = train_data
        valid_x1, valid_x2 = valid_data
        
        U_ph = tf.placeholder(tf.float32, [self.hiddens_dim[-1], self.output_dim])
        V_ph = tf.placeholder(tf.float32, [self.hiddens_dim[-1], self.output_dim])
        UtF = tf.matmul(tf.transpose(U), tf.transpose(f_out))
        GtV = tf.matmul(g_out, V)
        canon_corr = tf.trace(tf.matmul(UtF, GtV))
        update = optimizer(learning_rate).minimize(-canon_corr)
        
        for epoch_i in range(1, epochs+1):
            for batch_x1, batch_x2 in self._get_batch(train_x1, train_x2):
                f_output_ = self.sess.run(self.f_output, feed_dict={self.f_input: batch_x1})
                g_output_ = self.sess.run(self.g_output, feed_dict={self.g_input: batch_x2})
                U_, V_ = self.calc_CCA(f_output_, g_output_)
                self.U.assign(U_)
                self.V.assign(V_)
                self.sess.run(update, feed_dict={self.f_input: batch_x1,
                                                 self.g_input: batch_x2,
                                                 U_ph: U_,
                                                 V_ph: V_,
                                                 self.training: True})

            if epoch_i % display_size == 0:
                train_loss = self.sess.run(update, feed_dict={self.f_input: train_x1,
                                                              self.g_input: train_x2,
                                                              U_ph: U_,
                                                              V_ph: V_,
                                                              self.training: False})
                train_loss_history.append(train_loss)
                if valid_data is not None:
                     valid_loss = self.sess.run(update, feed_dict={self.f_input: valid_x1,
                                                                   self.g_input: valid_x2,
                                                                   U_ph: U_,
                                                                   V_ph: V_,
                                                                   self.training: False})
                    valid_loss_history.append(valid_loss)
                    print('Epoch {:>3}/{} Training loss: {:>6.3f}  - Validation loss: {:>6.3f}'.
                          format(epoch_i, epochs, train_loss, valid_loss))
                else:
                    print('Epoch {:>3}/{} Training loss: {:>6.3f}'.format(epoch_i, epochs, train_loss))
            if save_path and save_size and epoch_i % save_size == 0:
                self.save(save_path, epoch_i)
                
        return train_loss_history, valid_loss_history

    def predict(self, x1, x2, load_path=''):
        if load_path:
            self.load(load_path)
        
        x1_proj = self.sess.run(self.f_cca, feed_dict={self.f_input: x1, self.training: False})
        x2_proj = self.sess.run(self.f_cca, feed_dict={self.f_input: x2, self.training: False})
        return x1_proj, x2_proj
        
    def save(self, save_path, global_step=None):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path, global_step, write_meta_graph=False)
        print('Model save at ' + save_path + '-' + str(global_step))

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print('Model restored from' + path)
