
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.optimizers import Adam

def build_deep_model(input_shape, depth=3, filters=32, use_bilstm=False, lr=0.0001):
    """
    Xây dựng mô hình CNN với cấu hình động.
    Tên hàm chuẩn: build_deep_model (để khớp với trainer.py)
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    current_filters = filters  # Đổi tên biến này cho khớp trainer
    
    # === Vòng lặp tạo độ sâu (Depth) ===
    for i in range(depth):
        # Conv Block
        x = layers.Conv2D(current_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Tăng gấp đôi số filter sau mỗi lớp
        current_filters *= 2
    
    # === Classifier (Có hỗ trợ BiLSTM hoặc CNN thuần) ===
    if use_bilstm:
        # Nếu dùng BiLSTM: Reshape (H, W, C) -> (Time, Features)
        # Cách đơn giản: gộp H và W thành Time
        x = layers.Reshape((-1, current_filters))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    else:
        # Nếu dùng CNN thuần (Code cũ của bạn)
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    opt = Adam(learning_rate=lr)
    
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# Hàm dummy cho ML model (để tránh lỗi import nếu có)
def build_ml_model(algo='svm'):
    pass