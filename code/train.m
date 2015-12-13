function Model = train(data, labels)
    layerNum = 2;
    hiddenNum = 100;
    classNum = 10;
    eta = 0.1;
    maxIter = 300;
    cellSize = 4;
    convThresh = 0.02;
    
    [XTrain, YTrain] = nn_extract_feat(data, labels, cellSize);
    
    Model = nn_train(layerNum, hiddenNum, classNum, eta, maxIter, convThresh, XTrain, YTrain);
    save('Model.mat', 'Model');
end