setup;

% Load the CNN model
net = load(fullfile(pwd, 'data', 'mnistCNN.mat'));
% Display the network structure of the MNIST model
vl_simplenn_display(net);

% Load the test data
load(fullfile(pwd, 'data', 'MNIST.mat'), 'testLabel', 'testImages')
N = size(testImages, 3);
testLabelPredicted = zeros(N, 1);
for i=1:N
	% pick and image for classification
    im_  = single(testImages(:,:,i));
	% NO resizing needed
    %im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
	% remove mean
    im_ = im_ - net.normalization.averageImage ;

    % classify the image using CNN
    res = vl_simplenn(net, im_) ;

    % find the classification result
    scores = squeeze(gather(res(end).x));
    [bestScore, best] = max(scores);
    % subtract '1' as classes are labeled differently in CNN.
    testLabelPredicted(i, 1) = best-1;

% % %     To display the image and its classification result
% %     figure(1) ;
% %     imshow(im)
% %     set(gcf, 'name', sprintf('Class: %s, Score %.3f', net.classes.description{best}, bestScore));
end
test_error = sum(testLabelPredicted~=testLabel)/length(testLabel);
fprintf('Test error when using CNN: %1.2f%%\n', 100*test_error)