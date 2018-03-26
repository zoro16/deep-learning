from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, \
GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
# from training_data_generator import *
import numpy as np
from td_utils import *


# Load preprocessed training examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")

# Load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")

# x = graph_spectrogram("audio_examples/example_train.wav")

# _, data = wavfile.read("audio_examples/example_train.wav")
# print("Time steps in audio recording before spectrogram", data[:,0].shape)
# print("Time steps in input after spectrogram", x.shape)

Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375 # The number of time steps in the output of our model

def model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """    
    X_input = Input(shape = input_shape)

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, 15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization()(X)              # Batch normalization
    X = Activation('relu')(X)                # ReLu activation
    X = Dropout(0.8)(X)                      # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(128, return_sequences=True)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                      # dropout (use 0.8)
    X = BatchNormalization()(X)              # Batch normalization
    
    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(128, return_sequences=True)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                      # dropout (use 0.8)
    X = BatchNormalization()(X)              # Batch normalization
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)    
    return model  


model = model(input_shape = (Tx, n_freq))
model.summary()

model = load_model('./models/tr_model.h5')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

### PREDICTION ###

def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions

chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')


# filename = "./raw_data/dev/1.wav"
# prediction = detect_triggerword(filename)
# chime_on_activate(filename, prediction, 0.5)
# IPython.display.Audio("./chime_output.wav")

# filename  = "./raw_data/dev/2.wav"
# prediction = detect_triggerword(filename)
# chime_on_activate(filename, prediction, 0.5)
# IPython.display.Audio("./chime_output.wav")

