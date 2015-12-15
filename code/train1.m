function [Model1] = train1(X, Y)
    %% Extract HoG features
    cellSize = 4;

    XTrainFeat = [];
    
    for i = 1:size(X,1)
        im = im2single(reshape(X(i,:), 32, 32, 3));
        hog = vl_hog(im, cellSize);
        XTrainFeat = [XTrainFeat;hog(:)'];
    end
    %% Train on HoG features
    XTrainFeat = double(XTrainFeat);
    Y = double(Y);
    
    V = MyPCA(XTrainFeat, 100);
    XTrainFeatNew = XTrainFeat * V;
    
    [priors, classes, avg, stdev] = NaiveBayesClassifier(XTrainFeatNew, Y);
    field1 = 'priors';
    field2 = 'classes';
    field3 = 'avg';
    field4 = 'stdev';
    field5 = 'V';
    
    value1 = priors;
    value2 = classes;
    value3 = avg;
    value4 = stdev;
    value5 = V;
    
    Model1 = struct(field1, value1, field2, value2, field3, value3, field4, value4, field5, value5);
    save('Model1.mat', 'Model1');
end

function [priors, classes, avg, stdev] = NaiveBayesClassifier(XTrain, YTrain)
    
    [~, feature] = size(XTrain);

   
    classes = unique(YTrain);
    classNumber = numel(classes);
    priors = hist((YTrain),classes);
    priors = priors/sum(priors);
    
    avg = zeros(classNumber, feature);
    stdev = zeros(classNumber, feature);
    for c = 1 : classNumber
        avg(c,:) = mean(XTrain(YTrain == classes(c),:));
        stdev(c,:) = std(XTrain(YTrain == classes(c),:));
    end
end