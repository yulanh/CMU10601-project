function [YHat] = classify2(W, data)
    cellSize = 4;
    
    [X, ~] = lr_extract_feat(data, [], cellSize);
    [~,YHat] = max(X*W', [], 2);
    
    YHat = YHat - 1;
end