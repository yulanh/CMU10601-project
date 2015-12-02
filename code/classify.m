function YHat = classify(NNet, data)
% classifier using neural net, class [0, classNum-1]

    cellSize = 4;
    [X, ~] = nn_extract_feat(data, [], cellSize);
    NNOut = nn_forward(NNet, X);
    [~,YHat] = max(NNOut{end}, [], 2);
    
    YHat = YHat - 1;
end