from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

def train_text_only_model(X_train, y_train, preprocess, encoder):
    """Train text-only model."""
    # Inputs
    text_input = keras.Input(shape = (), dtype = tf.string, name = 'input')
    
    # Encoder layer
    preprocessed_text = hub.KerasLayer(preprocess)(text_input)
    encoder_outputs = hub.KerasLayer(encoder, trainable = True)(preprocessed_text)
    pooled_output = encoder_outputs['pooled_output']
    
    # Classification head
    x = Dense(128, activation = 'relu')(pooled_output)
    x = Dropout(0.5)(x)
    output = Dense(7, activation = 'softmax')(x)
    
    # Train model
    model = Model(inputs = [text_input], outputs = output)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(X_train['input'], y_train, batch_size = 16, epochs = 3, validation_split = 0.3)
        
    return model

def train_image_only_model(X_train, y_train):
    """Train image-only model."""
    # Inputs
    image_input = keras.Input(shape = (2048 * 3, ), dtype = tf.float32, name = 'input')
    
    # Dense layers and embedding extraction
    x = Dense(512, activation = 'relu')(image_input)
    x = Dropout(0.1)(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation = 'relu')(x)
    embeddings = Dropout(0.1)(x)
    image_model_for_training = Model(inputs = image_input, outputs = embeddings)
    
    # Classification head
    output = Dense(7, activation = 'softmax')(embeddings)

    # Train model
    image_model_for_classification = Model(inputs = image_input, outputs = output)
    image_model_for_classification.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    image_model_for_classification.fit(X_train, y_train, batch_size = 16, epochs = 3, validation_split = 0.3)

    return image_model_for_classification

def train_multi_modal_model(X_train, y_train, preprocess, encoder):

    # Text embeddings
    text_input = Input(shape = (), dtype = tf.string, name = 'text_input')
    preprocessed_text = hub.KerasLayer(preprocess)(text_input)
    encoder_outputs = hub.KerasLayer(encoder, trainable = True)(preprocess)
    pooled_output = encoder_outputs['pooled_output']
    text_embeddings = Dense(128, activation = 'relu')(pooled_output)
    text_embedding_model = Model(inputs = text_input, outputs = text_embeddings)
    
    # Image embeddings
    image_input = Input(shape = (2048 * 3,), dtype = tf.float32, name = 'image_input')
    image_embeddings = Dense(128, activation = 'relu')(image_input)
    image_embedding_model = Model(inputs = image_input, outputs = image_embeddings)
    
    # Concatenated embeddings
    concatenated_embeddings = Concatenate()([text_embedding_model.output, image_embedding_model.output])
    concatenated_embeddings = Lambda(lambda x: tf.math.l2_normalize(x, axis = 1))(concatenated_embeddings)
    
    # Classification head
    x = Dense(128, activation = 'relu')(concatenated_embeddings)
    x = Dropout(0.5)(x)
    final_output = Dense(7, activation = 'softmax')(x)
    
    # Train model
    model = Model(inputs = [text_input, image_input], outputs = final_output)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, y_train, batch_size = 32, epochs = 5, validation_split = 0.3)

    return model