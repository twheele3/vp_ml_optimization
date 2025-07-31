import tensorflow as tf

def BuildModel(imageSize=(256,256), 
               dropoutRate=0.15, 
               firstLayerFilterCount=8, 
               attn_layers=True, 
               bottom_attn=True, 
               depth = 5, 
               max_attn_layers = 1,
               inner_activation = 'linear',
               outer_activation = 'relu',
               upconv_kernel = 2,
               upconv_stride = 2,):
    # First layer in the network is each pixel in the grayscale image
    input1 = tf.keras.layers.Input((imageSize[0], imageSize[1], 1), name='image')
    input2 = tf.keras.layers.Input((7,),name = 'metadata')
    # token_embedding = tf.keras.layers.Embedding(input_dim=2**8, 
    #                                             output_dim=2**5, 
    #                                             input_length=1,
    #                                             name = f'token_embed')(input2)
    if attn_layers:
        inputDense = tf.keras.layers.BatchNormalization(name = 'token_bn')(input2)
        inputDense = tf.keras.layers.Dense(2**7,
                                           activation='linear',
                                           name='dense0')(inputDense)
        inputDense = tf.expand_dims(tf.expand_dims(inputDense,axis=-2),axis=-2)
    
    # Contracting path identifies features at increasing levels of detail
    # (i.e. intensity, edges, shapes, texture...)
    currentLayer = input1
    contractingLayers = []
    for i in range(depth):
        if i > 0:
            currentLayer = tf.keras.layers.MaxPooling2D((2, 2), name = f'maxpool{i}')(currentLayer)

        size = firstLayerFilterCount * 2 ** i
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation=inner_activation,
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              name=f'conv2d{i}_1')(currentLayer)
        currentLayer = tf.keras.layers.BatchNormalization()(currentLayer)
        currentLayer = tf.keras.layers.Activation("relu")(currentLayer)
        currentLayer = tf.keras.layers.Dropout(dropoutRate,
                                               name=f'dropout{i}')(currentLayer)
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation=inner_activation,
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              name=f'conv2d{i}_2')(currentLayer)
        currentLayer = tf.keras.layers.BatchNormalization()(currentLayer)
        currentLayer = tf.keras.layers.Activation(outer_activation)(currentLayer)
        contractingLayers.append(currentLayer)

    if attn_layers & bottom_attn:
        contractedPool = tf.keras.layers.MaxPooling2D(pool_size=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                                      strides=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                                      name = 'contractbottom')(currentLayer)
        contractedPool = tf.concat([contractedPool,inputDense],axis=-1, name='bottom_concat')
        contractedPool = tf.keras.layers.BatchNormalization()(contractedPool)
        contractedPool = tf.keras.layers.Dense(size,
                                               name = 'bottom_dense_1')(contractedPool)
        contractedPool = tf.keras.layers.Dropout(dropoutRate,
                                               name=f'upconv_attn_dropout{i}')(contractedPool)
        contractedPool = tf.keras.layers.Dense(size,
                                               name = 'bottom_dense_2')(contractedPool)
        contractedPool = tf.keras.layers.Activation('sigmoid')(contractedPool)
        currentLayer = currentLayer * contractedPool
    


    
    # Expanding path spatially places recognized features.
    for i in reversed(range(depth-1)):
        size = firstLayerFilterCount * 2 ** i

        currentLayer = tf.keras.layers.Conv2DTranspose(size, upconv_kernel, strides=upconv_stride,
                                                       padding='same',
                                                       name=f'upconv_transpose{i}')(currentLayer)
        # if upconv_stride == 1:
        #     currentLayer = tf.keras.layers.UpSampling2D(2)(currentLayer)
        # Removed as this isn't highlighting existing, but generating new from upconv
        currentLayer = tf.keras.layers.BatchNormalization()(currentLayer)
        currentLayer = tf.keras.layers.Activation(outer_activation)(currentLayer)
        currentLayer = tf.keras.layers.Concatenate(name=f'concat{i}')([currentLayer, contractingLayers[i]])

        if attn_layers:
            if type(max_attn_layers) == int:
                attn_layer_list = [i for i in reversed(range(depth-1))][:max_attn_layers]
            elif type(max_attn_layers) == list:
                attn_layer_list = max_attn_layers
            if i in attn_layer_list:
                contractedPool = tf.keras.layers.MaxPooling2D(pool_size=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                                          strides=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                                          name = f'upconv_pool{i}')(currentLayer)
                contractedPool = tf.keras.layers.BatchNormalization()(contractedPool)
                contractedPool = tf.concat([contractedPool,inputDense],axis=-1,name=f'upconv_concat{i}')
                contractedPool = tf.keras.layers.Dense(size*2,
                                                       name = f'upconv_attn{i}_1')(contractedPool)
                contractedPool = tf.keras.layers.Dropout(dropoutRate,
                                               name=f'upconv_attn_dropout{i}')(contractedPool)        
                contractedPool = tf.keras.layers.Dense(size*2,
                                                       name = f'upconv_attn{i}_2')(contractedPool)
                contractedPool = tf.keras.layers.Activation('sigmoid')(contractedPool)
                currentLayer = currentLayer * contractedPool
        
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation=inner_activation,
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              name=f'upconv_2d{i}_1')(currentLayer)
        currentLayer = tf.keras.layers.BatchNormalization()(currentLayer)
        currentLayer = tf.keras.layers.Dropout(dropoutRate,
                                               name=f'upconv_dropout{i}')(currentLayer)
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation=inner_activation,
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              name=f'upconv_2d{i}_2')(currentLayer)
        currentLayer = tf.keras.layers.BatchNormalization()(currentLayer)

    # Last layer is sigmoid to produce probability map.
    final = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name = 'output_shape')(currentLayer)

    model = tf.keras.Model(inputs=[input1,input2], outputs=[final])
    return model