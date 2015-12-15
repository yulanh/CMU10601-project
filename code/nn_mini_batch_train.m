function NNet = nn_mini_batch_train(layerNum, hidNodeNum, classNum, eta, maxIter, convThresh, XTrain, YTrain, batchSize)
    % layerNum -- including input, not output
    % classNum -- class should be [1, classNum]
    % eta -- step size for updating weights
    
    %%%%%%%%%%%
%     load ../subset_CIFAR10/small_data_batch_5
%     [XTest, YTest] = nn_extract_feat(data, labels, 4);
%     load ../cifar-10-batches-mat/data_batch_2.mat
%     data = data(1:1000,:);
%     labels = labels(1:1000,:);
%     [XTest, YTest] = nn_extract_feat(data, labels, 4);
    %%%%%%%%%%%
    
    fprintf('Layer: %d, Hidden Node: %d, Step Size: %.3f \n', layerNum, hidNodeNum, eta);
    
    %% initialize
    clear NNet;
    [trainNum, featSize] = size(XTrain);
%     preWeightNorm = 0;
%     preWeightNorm1 = 0;
    
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
    end_train = 0;
    if batchSize == -1
        batchSize = trainNum;
    end
    batchNum = trainNum / batchSize; % assume trainNum can be divided by batchSize
    
    for i = 1:maxIter
        fprintf('Iter %d/%d: ', i, maxIter);
        
        for b_i = 1:batchNum
            % mini-batch
            end_x_i = b_i * batchSize;
            begin_x_i = end_x_i - batchSize + 1;
            
            delta_w_acc = init_delta_w(classNum, layerNum, hidNodeNum, featSize);
            X = XTrain(begin_x_i:end_x_i,:);
            Y = YTrain(begin_x_i:end_x_i,:);
            NNOut = nn_forward(NNet, X);

            delta_w_acc = nn_backward_delta(NNet, X, Y, NNOut, classNum, layerNum, delta_w_acc);
            NNet = nn_backward_update(NNet, delta_w_acc, eta, layerNum, batchSize);
        
            delta_w = norm(delta_w_acc{layerNum}/batchSize);
            delta_w1 = norm(delta_w_acc{layerNum-1}/batchSize);
            
            % check convergence (using norm)
            if (delta_w < convThresh) && (delta_w1 < convThresh) 
                end_train = 1;
                break;
            end
        
        end
        
%         %%%%%%%%%%%%%%%%%%%
%         acc = get_acc(NNet, XTrain,YTrain);
%         acctest = get_acc(NNet, XTest,YTest);
%         fprintf('Train Accuracy: %.4f, Test Accuracy: %.4f, Delta Weight: %.4f, %.4f\n', acc, acctest, delta_w, delta_w1);
%         fprintf('Train Accuracy: %.4f, Delta Weight: %.4f, %.4f\n', acc, delta_w, delta_w1);
%         if acctest >= 0.54
%             break;
%         end
%         %%%%%%%%%%%%%%%%%%%

        if end_train == 1
            break;
        end
        
%         ii = mod(i,10);
%         save(strcat('NNModel', num2str(ii), '.mat'),'NNet');
    end
end

function delta_w = init_delta_w(classNum, layerNum, hidNodeNum, featSize)
    delta_w = cell(layerNum,1);
    
    % input layer ----- NNet(1): input l1, output l2
    delta_w{1} = zeros(featSize+1, hidNodeNum);
    
    % hidden layers
    for l = 2: 1: layerNum-1
        delta_w{l} = zeros(hidNodeNum + 1, hidNodeNum);
    end
    
    % output layer -- input layerNum, output layerNum+1(output layer)
    delta_w{layerNum} = zeros(hidNodeNum + 1, classNum);
end