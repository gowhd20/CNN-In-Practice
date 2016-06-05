clear
close all
imtool close all
clc

K           = 10; % max 'k' value which will be used while trainig
trainP      = 0.8; % proportion of training data which will be used to learn kNN.

%% Load training and test data
load('MNIST.mat', 'trainLabel', 'testLabel', 'trainImages', 'testImages')

%% Display Samples
%numDisplay = 20; % how many samples from each class to display
%viewMNISTData(trainImages, trainLabel, numDisplay)

%% Select a small subset of training data for quick experimentation: Remove (comment) this section when your code is working.
idx = randperm(length(trainLabel), 2000);% select 2000 samples randomly
trainImages = trainImages(:,:,idx);
trainLabel = trainLabel (idx);

% Find number of training & test images (N & Ntest)
%N = length(idx);
N = length(trainLabel);
Ntest = length(testImages(1,1,:));

%% Raw pixels Classifier
%% Prepare data for training and validation
% Reshape training & test images into vectors. Size of training and test data after reshaping will be 784*N  & 784*Ntest
trainVectors = reshape(trainImages, 28*28, N);
trainVectors = trainVectors';

testVectors = reshape(testImages, 28*28, Ntest);
testVectors = testVectors';

% Take transpose of the training & testing vectors so that each sample is in a row instead of a column (size will be N*784 & Ntest*784).

% Split data into Training & Validation sets using 'splitData'
[trainingData, trainingLabels, validationData, validationLabels] = ...
splitData(trainVectors, trainLabel, trainP);


%% Use Pixel values to train k-NN models for values of k from 1 to K
for k=1:K
    % Use trainingData to learn a kNN classifier
    
    % Use validationData and learnt kNN classifier to predict labels of validation data

    % Compute errors for validation data
    
    % Use trainingData and learnt kNN classifier to predict labels of training data

    % Compute errors for training data
    kNNPixels{k} = fitcknn(trainingData, trainingLabels, 'NumNeighbors', k); % classifiers

    valLabelsPred = predict(kNNPixels{k}, validationData);

    valErrorPixels(k) = sum(valLabelsPred~=validationLabels)/length(validationLabels);
    
end

%% Find best model & find its error on test data.
% Find best value of 'k' for which validation error is minimum.

% Use the model trained with best value of 'k' to classify test data

% Compute test error

minErrVal = min(valErrorPixels);
[,bestClfIdx] = find(valErrorPixels == minErrVal);
testLabelsPred = predict(kNNPixels{bestClfIdx}, testVectors); % predict values by the classifier with lowest rate of error 

%% Plot validation and training error

%% Find confusion matrix and display it
cmPixels = confusionmat(testLabelsPred, testLabel);
figure('name', 'Pixels: Confusion Matrix');
imshow(cmPixels, [], 'InitialMagnification', 'fit');

%% HOG Classifier
%% Compute HOG for training images
for i=1:N % N is the number of training samples
    % Initialize trainVectorsHOG
    if i==1
        % Find the size of HOG vector: HOG parameters can be changed here
        hogCellSize = [7 7];
        hogBins = 12;
        dim = length(extractHOGFeatures(trainImages(:,:,i), 'CellSize', hogCellSize, 'NumBins', hogBins));
        trainVectorsHOG = zeros(length(idx), dim, 'single');
    end
    trainVectorsHOG(i, :) = extractHOGFeatures(trainImages(:,:,i), 'CellSize', hogCellSize, 'NumBins', hogBins);
end

%% Compute HOG for testing images
for i=1:Ntest% Ntest is the number of testing samples
    if i==1
        testVectorsHOG = zeros(Ntest, dim, 'single');
    end
    testVectorsHOG(i, :) = extractHOGFeatures(testImages(:,:,i), 'CellSize', hogCellSize, 'NumBins', hogBins);    
end

%% Use HOG features instead of Pixel values to lean, validate and evaluate kNN classifiers as you did above.

[trainingDataHOG, trainingLabelsHOG, validationDataHOG, validationLabelsHOG] = ...
    splitData(trainVectorsHOG, trainLabel, trainP);

for k=1:K

    kNNPixelsHOG{k} = fitcknn(trainingDataHOG, trainingLabelsHOG, 'NumNeighbors', k); % classifiers

    valLabelsPredHOG = predict(kNNPixelsHOG{k}, validationDataHOG);

    valErrorPixelsHOG(k) = sum(valLabelsPredHOG~=validationLabelsHOG)/length(validationLabelsHOG);
    
end

minErrValHOG = min(valErrorPixelsHOG);
[,bestClfIdxHOG] = find(valErrorPixelsHOG == minErrValHOG);
testLabelsPredHOG = predict(kNNPixelsHOG{bestClfIdxHOG}, testVectorsHOG);

cmPixelsHOG = confusionmat(testLabelsPredHOG, testLabel);
figure('name', 'Pixels: Confusion Matrix HOG');
imshow(cmPixelsHOG, [], 'InitialMagnification', 'fit');

figure, plot(valErrorPixels), title('error value');
figure, plot(valErrorPixelsHOG), title('error value HOG');