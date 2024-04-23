from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class basicCNNPipeline:    
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.validation_dataset = None
        self.model = None
    
    def loadDataset(self, traindir, testdir, input_shape, split_size):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.1,
            validation_split=split_size
        )

        test_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        self.train_dataset = train_datagen.flow_from_directory(
            traindir,
            target_size=input_shape,
            color_mode='grayscale',
            class_mode='sparse',
            subset='training'
        )

        self.validation_dataset = train_datagen.flow_from_directory(
            traindir,
            target_size=input_shape,
            color_mode='grayscale',
            class_mode='sparse',
            subset='validation'
        )

        self.test_dataset = test_datagen.flow_from_directory(
            testdir,
            target_size=input_shape,
            batch_size=64,
            color_mode='grayscale',
            class_mode='sparse'
        )
    
    def loadModel(self, iput_shape):
        inputs = tf.keras.Input(shape=(iput_shape))#(28, 28, 1))

        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool1)
        maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

        flatten = tf.keras.layers.Flatten()(maxpool2)

        dense1 = tf.keras.layers.Dense(256, activation='relu')(flatten)
        dense2 = tf.keras.layers.Dense(256, activation='relu')(dense1)

        outputs = tf.keras.layers.Dense(24, activation='softmax')(dense2)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def compileModel(self):
        if self.model:
            self.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

    def getModelSummary(self):
        summary = "Model not found."
        if self.model:
            summary = self.model.summary()
        return summary
    
    def trainModel(self, epochs=5):
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.validation_dataset)
    
    def evaluateModel(self):
        self.model.evaluate(self.test_dataset)
    
    def getModel(self):
        return self.model
