import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.optimizers import Adam

def build_deep_model(input_shape, depth=3, filters=32, use_bilstm=False, lr=0.0001):
    """
    Build a configurable CNN model with optional BiLSTM head.

    Args:
        input_shape: Input tensor shape (H, W, C).
        depth:       Number of Conv blocks; each block doubles the filter count.
        filters:     Number of filters in the first Conv block.
        use_bilstm:  If True, replace GlobalAveragePooling with a BiLSTM layer.
        lr:          Learning rate for the Adam optimizer.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    current_filters = filters

    # Build convolutional blocks; filter count doubles with each block
    for i in range(depth):
        x = layers.Conv2D(current_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        current_filters *= 2

    # Sequence modeling head (BiLSTM) or spatial pooling head (GAP)
    if use_bilstm:
        x = layers.Reshape((-1, current_filters))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_ml_model(algo='svm'):
    pass