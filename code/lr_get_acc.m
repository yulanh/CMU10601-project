function acc = lr_get_acc(W, X, Y)

    YHat = classify2(W, X);
    Y = double(Y);
    
    cMat = confusionmat(Y, YHat);
    acc = sum(diag(cMat))/sum(sum(cMat));  
end