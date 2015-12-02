function acc = nn_get_acc(NNet, X, Y)
    
    YHat = classify(NNet, X);
    Y = double(Y);
    
    cMat = confusionmat(Y, YHat);
    acc = sum(diag(cMat))/sum(sum(cMat));  
end