function Model = train(data, labels)
    layerNum = 2;
    hiddenNum = 60;
    classNum = 10;
    eta = 0.001;
    maxIter = 1000;
    cellSize = 4;
    convThresh = 0.01;
    batchSize = 10;
    [XTrain, YTrain] = nn_extract_feat(data, labels, cellSize);
%     save('Feature5000.mat','XTrain','YTrain');
%     load('Feature5000.mat');
    
    Model = nn_mini_batch_train(layerNum, hiddenNum, classNum, eta, maxIter, convThresh, XTrain, YTrain, batchSize);
    save('Model.mat', 'Model');
end