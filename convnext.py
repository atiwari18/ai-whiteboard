import numpy
from tf_keras.applications import ConvNeXtBase
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.layers import Dense, GlobalAveragePooling2D
from tf_keras.models import Sequential
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, accuracy_score
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainDatagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
testDatagen = ImageDataGenerator(rescale=1./255)

trainSet = trainDatagen.flow_from_directory(
    'dataset',
    target_size=(160, 160),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

valSet = trainDatagen.flow_from_directory(
    'dataset',
    target_size=(160, 160),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

testSet = testDatagen.flow_from_directory(
    'test',
    target_size=(160, 160),
    batch_size=8,
    class_mode='categorical',
    shuffle=False  # Important for evaluation
)

baseModel = ConvNeXtBase(weights='imagenet', include_top=False, input_shape=(160, 160, 3)) #builds model

model = Sequential([
    baseModel,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #compiles model

checkpoint = ModelCheckpoint('convnext_gesture_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
earlyStop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)

history = model.fit(
    trainSet,
    validation_data=valSet,
    epochs=1,
    callbacks=[checkpoint, earlyStop]
) #trains model

testLoss, testAcc = model.evaluate(testSet) #evals model on test set
print(f'Test Accuracy: {testAcc:.4f}')

yPred = model.predict(testSet)
yPredClasses = numpy.argmax(yPred, axis=1)

yT = testSet.classes

print(classification_report(yT, yPredClasses, target_names=testSet.class_indices.keys(), zero_division=1))

print('Overall Accuracy:', accuracy_score(yT, yPredClasses))