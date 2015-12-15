function W = train2(XTrain, YTrain)
    classNum = 10;
    eta = 1e-5;
    lambda = 10;
    maxIter = 5000;
    w0Range = 1e-3;
    deltaThresh = 1e-5;%1e-4;
    cellSize = 4;
    
%     [XTrain,YTrain] = lr_extract_feat(XTrain,YTrain,cellSize);
%     save('lrFeature5000.mat','XTrain','YTrain');
    load 'lrFeature5000.mat'
    
    
    % Initialize W and prepare Y 
    [dataNum,D] = size(XTrain);
    W = w0Range * rand(classNum, D);
    Y = zeros(dataNum, classNum);
    
    for class = 1:classNum
        Y(YTrain==class, class) = 1; 
    end
    
    % Train Logistic Regression
    Model = lr_train(W, XTrain, Y, classNum, maxIter, lambda, eta, deltaThresh);
    save('Model2.mat', 'Model');
end

function W = lr_train(W, X, Y, classNum, maxIter, lambda, eta, deltaThresh)
    %%%%%%%%%%%
    load ../cifar-10-batches-mat/data_batch_2.mat
    XTest = data(1:1000,:);
    YTest = labels(1:1000);
    [XTest,YTest] = lr_extract_feat(XTest,YTest,4);
    load 'lrFeature5000.mat'
    %%%%%%%%%%%
    
    % set up prior (not bias term)
    lambda = lambda * ones(size(W));
    lambda(:,1) = 0;
    
    preYVal = 0;
    for i = 1:maxIter
        fVal = X * W'; % N*C
        fVal = exp(fVal - repmat(max(fVal, [], 2), [1,classNum]));
        fVal = fVal ./ repmat(sum(fVal, 2), [1,classNum]);

        yVal = - sum(sum(Y .* log(fVal))) + 0.5 * sum(sum(lambda .* (W .^ 2)));
        yGrad = - (Y - fVal)' * X + lambda .* W;
        
        W = W - eta * yGrad;
        
        deltaYVal = abs((yVal - preYVal) / preYVal);
        if deltaYVal <= deltaThresh
            break;
        end
        preYVal =  yVal;
        
        %%%%%%%%%%%
        fprintf('Iter: %d; Train: %.4f, Test: %.4f\n', i, lr_acc(W, XTrain, YTrain), lr_acc(W, XTest, YTest));
%         fprintf('Iter: %d; Test Accuracy: %.4f\n', i, lr_acc(W, XTest, YTest));
        %%%%%%%%%%%
    end
end

function res = lr_acc(W, X, Y)

    [~,YHat] = max(X*W', [], 2);
    res = sum(Y==YHat) / length(Y);
    
end