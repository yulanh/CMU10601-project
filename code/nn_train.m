function NNet = nn_train(layerNum, hidNodeNum, classNum, eta, maxIter, XTrain, YTrain)
    % layerNum -- including input, not output
    % classNum -- class should be [1, classNum]
    % eta -- step size for updating weights
    
%     %%%%%%%%%%%
%     load ../subset_CIFAR10/small_data_batch_5
%     XTest = data;
%     YTest = labels;
%     [XTest, YTest] = nn_extract_feat(data, labels, 4);
%     %%%%%%%%%%%
    
    fprintf('Layer: %d, Hidden Node: %d, Step Size: %.3f \n', layerNum, hidNodeNum, eta);
    
    %% initialize
    clear NNet;
    [trainNum, featSize] = size(XTrain);
    
    % input layer ----- NNet(1): input l1, output l2
    NNet = struct('w', eye(featSize+1, hidNodeNum), 'f', 'sigmoid');
    
    % hidden layers
    for l = 2: 1: layerNum-1
        NNet(l).w = 0.1 * rand(hidNodeNum + 1, hidNodeNum); %NNet(l).w = rand(hidNodeNum+1, hidNodeNum)*2*epsilon-epsilon;
        NNet(l).f = 'sigmoid';
    end
    
    % output layer -- input layerNum, output layerNum+1(output layer)
    NNet(layerNum).w = 0.1 * rand(hidNodeNum + 1, classNum);
    NNet(layerNum).f = 'softmax';

    %% train
    for i = 1:maxIter
        fprintf('Iter %d/%d: ', i, maxIter);
        for x_i = 1:trainNum
            X = XTrain(x_i,:);
            Y = YTrain(x_i,:);
            NNOut = nn_forward(NNet, X);

            NNet = nn_backward(NNet, X, Y, NNOut, eta, classNum, layerNum);
        end
        
%         
%         %%%%%%%%%%%%%%%%%%%
%         acc = get_acc(NNet, XTrain,YTrain);
%         acctest = get_acc(NNet, XTest,YTest);
%         fprintf('Train Accuracy: %.4f, Test Accuracy: %.4f\n', acc, acctest);
%         %%%%%%%%%%%%%%%%%%%
        % TODO: check convergence (using norm ??)
    end
end
