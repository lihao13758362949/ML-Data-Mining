# 输入：splits,metrics,X_test
# 0 关于keras
# Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。Keras 的开发重点是支持快速的实验。能够以最小的时延把你的想法转换为实验结果，是做好研究的关键。
# 可以直接调用keras：`import keras`或`from keras.___ import ___`
# 也可以在tensorflow中调用keras：`tf.keras`

# 1. 指定模型
## 1.1 Sequential模型
from keras.models import Sequential # 顺序模型
from keras.models import Model # 自定义模型
# model = Model(inputs=X_input,outputs=y_output,name='Convnet')


## 1.2 模型搭建 # 输出model

# （1）model.summary() ：model总结，看看model建成什么样了

# （2）from keras.layers import Input # 输入层
# （3）from keras.layers import Dense # 全连接层
# Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# kernel_initializer, bias_initializer：初始化器，'zeros'，'glorot_uniform'
## 也可以显式指定
## from keras.initializers import glorot_uniform,glorot_normal
## kernel_initializer=glorot_uniform(seed=0)
# units：输出神经元个数。
# （4）from keras.layers import Activation # 激活函数层，你可以选择在其他层加入activation参数，也可以调用Activation层来指定激活函数,relu 或 softmax：多类别sigmid 或 sigmoid：二分类
# relu(x, alpha=0.0, max_value=None, threshold=0.0)
# tanh(x)
# sigmoid(x)
# linear(x)：缺省时采用这个。
# softmax(x, axis=-1)
# （5）from keras.layers.core import Reshape # 转换shape
# Reshape((3, 4), input_shape=(12,)) 
# （6） from keras.layers import Add
# （7） from keras.layers import RepeatVector
# RepeatVector(3) 从(None, 32)到(None, 3, 32)
# （8） from keras.layers import Multiply
# outputs = Multiply()([inputs, outputs])
from keras.layers import ZeroPadding2D, BatchNormalization, 
from keras.layers import concatenate # 一个张量，所有输入张量通过 axis 轴串联起来的输出张量。
# X = concatenate([X1,X2,X3],axis=3)
from keras.layers import Flatten # 铺平数据层
from keras.layers import Dropout # Dropout层 Dropout(rate, noise_shape=None, seed=None) 随机删除层中的若干神经元来避免过拟合。
# rate: 在 0 和 1 之间浮动。需要丢弃的输入比例。
# noise_shape: 1D 整数张量， 表示将与输入相乘的二进制 dropout 掩层的形状。 例如，如果你的输入尺寸为 (batch_size, timesteps, features)，然后 你希望 dropout 掩层在所有时间步都是一样的， 你可以使用 noise_shape=(batch_size, 1, features)。
# seed: 一个作为随机种子的 Python 整数。
from keras.layers import Conv2D # 二维卷积层（如图像上的空间卷积）。卷积运算的主要目的是使原信号特征增强,并降低噪音
# Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# filters: 整数，过滤器个数，也是输出的维数。
# kernel_size: 单个整数（两个数相同的略写）或两个整数组成的元组或列表，来指定卷积窗口的大小。
# input_shape：当该层作为第一层时要加入输入维度，如input_shape=(28,28,3)。
# strides: 指定卷积步幅，同样可以用一个整数代替。
# padding: 和tensorflow中一样，valid表示不加边，same表示加边到卷积后大小相同。
# data_format: A string, one of “channels_last” or “channels_first”. The ordering of the dimensions in the inputs. “channels_last” corresponds to inputs with shape (batch, height, width, channels) while “channels_first” corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be “channels_last”.
# dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
# use_bias: Boolean, whether the layer uses a bias vector.
# kernel_initializer: Initializer for the kernel weights matrix (see initializers).
# bias_initializer: Initializer for the bias vector (see initializers).
# kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
# bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
# activity_regularizer: Regularizer function applied to the output of the layer (its “activation”). (see regularizer).
# kernel_constraint: Constraint function applied to the kernel matrix (see constraints).
# bias_constraint: Constraint function applied to the bias vector (see constraints).
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D # 池化层
# MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# pool_size：指定池化窗口大小，如(2,2)，也可以只用一个2表示(2,2)。

### 1.2.1 add法
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

### 1.2.2 列表传递法
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

### 1.2.3 
def Convnet(input_shape=(187,1,1),classes=2):

    X_input = Input(input_shape)
    # X = ZeroPadding2D((10,1))(X_input)
    X1 = Conv2D(64,(11,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X_input)
    # X1 = BatchNormalization(axis=3)(X1)
    X1 = Activation('relu')(X1)
    X1 = MaxPooling2D((10,1),strides=(10,1))(X1)
    X2 = Conv2D(64,(21,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X_input)
    # X2 = BatchNormalization(axis=3)(X2)
    X2 = Activation('relu')(X2)
    X2 = MaxPooling2D((10,1),strides=(10,1))(X2)
    X3 = Conv2D(64,(31,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X_input)
    # X3 = BatchNormalization(axis=3)(X3)
    X3 = Activation('relu')(X3)
    X3 = MaxPooling2D((10,1),strides=(10,1))(X3)
    X = concatenate([X1,X2,X3],axis=3)
    X = Conv2D(64,(11,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,1),strides=(2,1))(X)
    X = Conv2D(64,(11,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,1),strides=(2,1))(X)
    X = Conv2D(64,(11,1),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,1),strides=(2,1))(X)
    X = Flatten()(X)
    X = Dense(512, kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(512, kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs=X_input,outputs=X,name='Convnet')

    return model

### 1.2.4
TIME_PERIODS = 187
def build_model(input_shape=(TIME_PERIODS,),num_classes=2):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, 1), input_shape=input_shape))
    model.add(Conv1D(16, 8,strides=2, activation='relu',input_shape=(TIME_PERIODS,1)))

    model.add(Conv1D(16, 8,strides=2, activation='relu',padding="same"))
#     model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(64, 4,strides=2, activation='relu',padding="same"))
#     model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(256, 4,strides=2, activation='relu',padding="same"))
#     model.add(MaxPooling1D(2))
    model.add(Conv1D(512, 2,strides=1, activation='relu',padding="same"))
    model.add(Conv1D(512, 2,strides=1, activation='relu',padding="same"))
#     model.add(MaxPooling1D(2))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return(model)
model = build_model()


## 1.3 配置学习过程 模型编译 输入：model，loss_function,optimizer,metrics 输出：model
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# 优化器 optimizer：如 rmsprop 或 adagrad 或 sgd，也可以是 Optimizer 类的实例。
## from keras.optimizers import RMSprop
### optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
## SGD(lr=0.01, momentum=0.9, nesterov=True) 
## Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)


# 损失函数 loss：categorical_crossentropy 或 mse或mean_squared_error 或 binary_crossentropy，也可以是一个目标函数
# 评估标准 metrics：accuracy，也可以是自定义的评估标准函数
# model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
              
# 自定义评估标准函数
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

# model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
              
## 1.4 训练过程 输入：model，callbacks回调函数，epochs，verbose， 输出：history训练历史，
### 1.4.1 fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
#### 1.4.1.1 多分类
# 将标签列转换为分类的 one-hot 编码
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10, dtype='float32')
history = model.fit(data, one_hot_labels, epochs=10, batch_size=32)

#### 1.4.1.2 二分类
history = model.fit(x_train, y_train, epochs=5, batch_size=32)
# model.train_on_batch(x_batch, y_batch)

#### 1.4.1.3 回调函数callbacks
from keras.callbacks import ReduceLROnPlateau # 当标准评估停止提升时，降低学习速率。
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
## 1.5 模型评估
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

from keras.metrics import accuracy
accuracy(y_true, y_pred)
### 1.5.1  Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

## 1.6 Display some error results 

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

## 1.6 模型预测
classes = model.predict(x_test, batch_size=128)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

# 10 .preprocessing
## 10.1 .image
1. `.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format='channels_last', validation_split=0.0, interpolation_order=1, dtype='float32')`
>`featurewise_center`：按特征将数据集上的输入平均值设置为0
>`samplewise_center`：按样本将数据集上的输入平均值设置为0
>`featurewise_std_normalization`：按特征除以标准差
>`samplewise_std_normalization`：按样本除以标准差
>`zca_whitening`：
>`rotation_range`：随机旋转角度最大范围(degrees, 0 to 180)
>`zoom_range`：随机缩放范围，可以给出区间[lower,upper]，也可以给一个float，表示[1-float,1+float]
>`width_shift_range`：水平偏移范围
>`height_shift_range`：垂直偏移范围
>`horizontal_flip`：水平翻转
>`vertical_flip`：垂直翻转
```py
# 数据增强
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
```
