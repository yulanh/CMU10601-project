function [Y] = classify(Model, X)
    
    cellSize = 4;
    XTestFeat = [];
    for i = 1:size(X,1)
        im = im2single(reshape(X(i,:), 32, 32, 3));
        hog = vl_hog(im, cellSize);
        XTestFeat = [XTestFeat;hog(:)'];
    end

    priors = Model.priors;
    classes = Model.classes;
    avg = Model.avg;
    stdev = Model.stdev;
    V = Model.V;
    
%   classify
    [testNumber, ~] = size(X);
    Y = zeros(testNumber, 1);
    classNumber = numel(classes);
    XTest = XTestFeat * V;
    
    for t = 1 : testNumber
        probClass = zeros(classNumber);
        for c = 1:classNumber
            prob = log(normpdf(XTest(t,:),avg(c,:),stdev(c,:)));
            probClass(c) = log(priors(c)) + sum(prob);
        end
        
        %Assign the class with the highest posterior probability
        [~, class_index] = max(probClass( : , 1));
        Y(t) = class_index - 1;
    end
end

