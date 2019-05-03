clc
clear all
close all
%% Set flags to true to train the network
train= true;

% Set this flags to plot
show.wrong_classified   = true;% wrong classified images
show.filter             = true;% filters(weights)
show.feature_maps       = true;

%% Datasetpath
matlabroot='dataset'
Datasetpath=fullfile(matlabroot)

%% Create an ImageDataStore For Loading Images
ds = imageDatastore(Datasetpath,...
    'IncludeSubfolders',...
    true,'FileExtensions','.png','LabelSource','foldernames')

labelCount = countEachLabel(ds)
%Split Data in the training folder into training and validation set
[train_set,val_set] = splitEachLabel(ds,0.8,0.2)


%% Image Augmenter
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-50,5], ...%
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3],...
    'RandXScale',scaleRange)

%% Resize the images for uniformity purposes
imageSize = [50 50 1];
augims = augmentedImageDatastore(imageSize,train_set,'DataAugmentation',imageAugmenter);
%% Visualize augmented some of the augmented images some of the

ims = augims.preview();
montage(ims{1:4,1})
%% Resize the validation set images to match the dimensions of the network's input layer
val_augmids=augmentedImageDatastore(imageSize,val_set)


%% Deep CNN Architecture
%% CNN layer array
layers = [
    imageInputLayer(imageSize,'Name','input_layer')
    convolution2dLayer(3,16,'Padding','same','Name','C1')
    batchNormalizationLayer('Name','B1')
    reluLayer('Name','R1')
    maxPooling2dLayer(2,'Stride',2,'Name','M1')
    dropoutLayer('Name','D1')
    
   
    convolution2dLayer(3,32,'Padding','same','Name','C2')
    batchNormalizationLayer('Name','B2')
    reluLayer('Name','R2')
    maxPooling2dLayer(2,'Stride',2,'Name','M2')
    

    convolution2dLayer(3,64,'Padding','same','Name','C3')
    batchNormalizationLayer('Name','B3')
    reluLayer('Name','R3')
    maxPooling2dLayer(2,'Stride',4,'Name','M3')
   
    
    %% Point of interconnection between the sequential Layer and the parallel convolution Layer
    additionLayer(2,'Name','add')
    
    
    fullyConnectedLayer(5,'Name','fc')
    %% Softmax Classifier
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')];

%% Directed Asysclic  Graph for the CNN
Dgraph = layerGraph(layers);
addConv= convolution2dLayer(3,64,'Stride',8,'Name','skipConv')
%% Add the  the 
Dgraph = addLayers(Dgraph,addConv);
%% Display the connection of the layers
Dgraph = connectLayers(Dgraph,'D1','skipConv');
Dgraph = connectLayers(Dgraph,'skipConv','add/in2');
figure
plot(Dgraph);


%% Training Hyperparameters

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.01, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MiniBatchSize',40, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',val_augmids, ...
    'ValidationFrequency',5, ...
    'Verbose',true, ...
    'Plots','training-progress');

%% Train The Network
network = trainNetwork(augims, Dgraph,options)

%% Save the trained Network

save network
%% TEST THE MODEL

matlabroot='testdata'% Path for test data
Datasetpath=fullfile(matlabroot)
%% Image Datastore for test images
testimds = imageDatastore(Datasetpath,'IncludeSubfolders',true,'Labelsource','foldernames');
labelCount = countEachLabel(testimds)
%% Resize Test Images to a constant Size
testimds.ReadSize = numpartitions(testimds)
testimds.ReadFcn = @(loc)imresize(imread(loc),[50,50])

%% Calculate Accuracy of the Model
[Prediction,scores] = classify(network,testimds)
Val_Labels = testimds.Labels;
accuracy = sum(Prediction == Val_Labels)/numel(Val_Labels)
figure(2)
plotconfusion(Val_Labels,Prediction)

%% Show wrongly classified images
if show.wrong_classified
    % % Find wrong classified examples
    imdx = find(testimds.Labels~=Prediction);
    imdx2 = int16(rand(length(imdx),1)*size(Prediction,1));
    fprintf('We have %d/%d wrong classifications\n',length(imdx),size(Prediction,1));
    for i = 1:length(imdx)
        img = readimage(testimds,double(imdx(i)));
        figh=figure(3);
        clf;
        set(figh,'Outerposition',[1,41,450,360]);
        subplot(1,2,1)
        colormap(gray);
        imagesc(img),
        axis square
        axis xy
        lab = sprintf('classified as %s',Prediction(imdx(i)));
        title(lab,'Color','Red');
        pause
    end
end


%% View activation of the first Convolutional Layer
figure(4)
image = imread('test.png');%% Load Images whose activation are to be investigated
image = imresize(image,network.Layers(1).InputSize(1:2));% Resize this image to the network size
imshow(image)  %%Display the image
activation = activations(network,image,'C1','OutputAs','channels');%% Activatin of the first convolution layer
size(activation) %Activation size
activation = reshape(activation,size(activation,1),size(activation,2),1,size(activation,3)); 
act_scaled = mat2gray(activation);
tmpact = act_scaled(:);
tmpact = imadjust(tmpact,stretchlim(tmpact));
stecthed_ac = reshape(tmpact,size(act_scaled));
figure(5)
montage(stecthed_ac)
title('Activations from the First Convolutional  layer','Interpreter','none')


%% Activations of the First ReLU Layer
activation = activations(network,image,'R1','OutputAs','channels');
size(activation)
activation = reshape(activation,size(activation,1),size(activation,2),1,size(activation,3));%Reshape into a 4D array
act_scaled = mat2gray(activation);% Convert matrix to grayscale image
tmpact = act_scaled(:);
tmpact = imadjust(tmpact,stretchlim(tmpact));
stecthed_ac = reshape(tmpact,size(act_scaled));
figure(6)
montage(stecthed_ac)
title('Activations from the The First Relu layer','Interpreter','none')

%% plot the filter weights of the first Convolution layer

if show.filter
    figh=figure(10);
    clf;%Delete from the current figure all graphics
    set(figh,'Outerposition',[451,41,450,360]);
    %Plot the Weight Matrices of first convolution layer Conv
    for i=1:16
        subplot(4,4,i)
        imagesc(network.Layers(2).Weights(:,:,1,i))
        set(gca, 'XTickLabel', [])
        set(gca, 'YTickLabel', [])
        axis xy
        colormap(gray);
    end
    subplot(4,4,2)
    title('First convolution Layer, Weights of filter 1:16');
end


