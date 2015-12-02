function Model = train(data, labels)
    layerNum = 2;
    hiddenNum = 100;
    classNum = 10;
    eta = 0.1;
    maxIter = 40;
    cellSize = 4;
    
    [XTrain, YTrain] = nn_extract_feat(data, labels, cellSize);
    
    Model = nn_train(layerNum, hiddenNum, classNum, eta, maxIter, XTrain, YTrain);
    save('Model.mat', 'Model');
end