at first I made a dataset with the help thse

This dataset contains total of 3000 annotated mosquito image from the sources

1. https://www.mosquitoalert.com/en/mosquito-images-dataset/

2. https://data.mendeley.com/datasets/88s6fvgg2p/4

3. https://ieee-dataport.org/open-access/image-dataset-aedes-and-culex-mosquito-species

4. https://datadryad.org/stash/dataset/doi:10.5061/dryad.z08kprr92

from these images we meticulously selected the best ones for our purpose. the mosquitos that are included are 

1. ANNOPHELES : 1000 images.

2. AEDES : 1000 images.

3. CULEX : 1000 images.

then I used transfer learning method .it contains baseline models analysis with thse





DenseNet121.ipynb     EfficientNetB2.ipynb     NasNet.ipynb     VGG19.ipynb

DenseNet169.ipynb     InceptionResNetV2.ipynb  ResNet101.ipynb  Xception.ipynb

DenseNet201.ipynb     InceptionV3.ipynb        ResNet152.ipynb

EfficientNetB0.ipynb  MobileNet.ipynb          ResNet50.ipynb

EfficientNetB1.ipynb  MobileNetV2.ipynb        VGG16.ipynb

and then thruough these analysis i came up with my new mdoel MosQNet-SA

which has this architecture


# Initial Conv Layer
x = Conv2D(32, (3, 3), padding='same', kernel_initializer=he_normal())(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)

# Residual Block
x = residual_block(x, filters=64)

# Inception-like Block
x = inception_block(x, filters=32)

# MBConv Block
x = MBConvBlock(x, expansion_ratio=6, output_filters=64, kernel_size=(3, 3), strides=(1, 1), se_ratio=0.25)

# Residual Block
x = residual_block(x, filters=32)

# Inception-like Block
x = inception_block(x, filters=16)

# MBConv Block
x = MBConvBlock(x, expansion_ratio=6, output_filters=32, kernel_size=(3, 3), strides=(1, 1), se_ratio=0.25)

# Spatial Attention Block
x = spatial_attention_block(x)


# Global Average Pooling
x = GlobalAveragePooling2D()(x)

# Dense layers
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.2)(x)

# Output layer (assuming 'folders' contains class labels)
prediction = Dense(len(folders), activation='softmax')(x)

# Create the final model
model = Model(inputs=inputs, outputs=prediction)

# Print the model summary
model.summary()


this is the output
Model	Optimizer	Learning Rate	Epochs	Total Params	Trainable Params	TP Percentage
VGG16	SGD	0.001	121	16846915	4491267	26.66
VGG19	SGD	0.001	81	22156611	6851075	30.92
ResNet50	SGD	0.001	75	32011395	9477635	29.61
ResNet101	SGD	0.001	53	51081859	16304131	31.92
ResNet152	SGD	0.001	59	66794627	20512259	30.71
Xception	ADAM	0.001	51	29285163	8427011	28.78
InceptionV3	ADAM	0.001	62	23935011	7287491	30.45
InceptionResNetV2	ADAM	0.001	68	55944675	12413027	22.19
MobileNet	RMSprop	0.001	73	7458243	4230659	56.72
MobileNetV2	RMSprop	0.001	67	7535939	5277187	70.03
DenseNet121	ADAM	0.001	80	11266883	4267523	37.88
DenseNet169	ADAM	0.001	38	19493699	6890243	35.35
DenseNet201	ADAM	0.001	61	26221379	8185027	31.22
NASNetMobile	ADAM	0.001	43	8630167	4359683	50.52
EfficientNetB0	ADAM	0.001	67	9327526	5279747	56.60
EfficientNetB1	ADAM	0.001	57	11853194	5689347	48.00
EfficientNetB2	ADAM	0.001	46	13570812	7417947	54.66
						
						
MosQNetSA	ADAM	0.001	81	388349	384155	98.92

Model	Training Loss	Validation Loss	Test Loss	Train Accuracy	Validation Accuracy	Test Accuracy
VGG16	0.12	0.16	0.2	95.88%	93.62%	93.37%
VGG19	0.25	0.30	0.3	91.21%	88.70%	88.76%
ResNet50	0.15	0.21	0.2	95.01%	91.01%	91.64%
ResNet101	0.20	0.24	0.3	92.95%	90.14%	89.34%
ResNet152	0.20	0.26	0.3	93.38%	90.14%	90.49%
Xception	0.05	0.17	0.22	98.12%	92.46%	93.37%
InceptionV3	0.02	0.16	0.23	99.28%	95.07%	94.81%
InceptionResNetV2	0.0075	0.1121	0.053	99.71%	97.10%	98.56%
MobileNet	0.01	0.10	0.15	99.78%	97.10%	97.12%
MobileNetV2	0.0336	0.1721	0.1191	98.92%	94.49%	97.41%
DenseNet121	0.01	0.0667	0.0845	99.78%	97.68%	97.69%
DenseNet169	0.0359	0.1065	0.0589	98.77%	96.23%	97.98%
DenseNet201	0.0015	0.0574	0.0349	99.96%	98.84%	99.14%
NASNetMobile	0.1102	0.2317	0.2037	95.84%	92.17%	93.95%
EfficientNetB0	0.0065	0.0613	0.077	99.78%	97.10%	97.41%
EfficientNetB1	0.0026	0.0722	0.0838	100.00%	98.55%	97.41%
EfficientNetB2	0.0069	0.0811	0.0732	99.89%	98.26%	97.69%
						
						
MosQNetSA	0.0189	0.0783	0.0401	99.78%	98.26%	99.42%

Metric	Class 1	Class 2	Class 3
Accuracy	0.987752	0.998721	0.973467
Precision	0.993664	0.990643	0.979466
Recall	0.980208	0.99882	0.985537
F1-score	0.98689	0.994715	0.982492

Augmentation Techniques	Description
Width Shift Range	Shift the width by up to 10%
Height Shift Range	Shift the height by up to 10%  
Shear Range	Apply shear transformation by 10%
Zoom Range	Apply zoom transformation by 10%
Horizontal Flip	Enable horizontal flipping
Fill Mode	Nearest neighbor filling mode
Preprocessing Function	Applies the 'preprocess_input' function to the image

Callback	Description
EarlyStopping	Stops training if validation loss does not improve for 20 epochs to prevent overfitting.
ModelCheckpoint	Saves the best model (based on validation loss) to 'best_model.h5'.
TensorBoard	Logs training/validation metrics and model architecture to TensorBoard.
ReduceLROnPlateau	Reduces learning rate by 0.2 if validation loss does not improve for 10 epochs, with a minimum of 1e-6.
CSVLogger	Logs training/validation metrics to 'training.log' CSV file.
LearningRateScheduler	Reduces learning rate by 0.9 every 10 epochs during training.


then I also used XAI tools like these

explainers = [
    Saliency(model),
    GradientInput(model),
    GuidedBackprop(model),
    IntegratedGradients(model, steps=80, batch_size=batch_size),
    SmoothGrad(model, nb_samples=80, batch_size=batch_size),
    SquareGrad(model, nb_samples=80, batch_size=batch_size),
    VarGrad(model, nb_samples=80, batch_size=batch_size),
    GradCAM(model),
    Occlusion(model, patch_size=10, patch_stride=5, batch_size=batch_size),
    Rise(model, nb_samples=300, batch_size=batch_size),
    SobolAttributionMethod(model, batch_size=batch_size),
    Lime(model, nb_samples=100),
    KernelShap(model, nb_samples=100)
]


I also have some figure in figures directory use those.

make me a github readme.md content of my paper... 




