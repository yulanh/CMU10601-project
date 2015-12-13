function delta_w_acc = nn_backward_delta(NNet, X, Y, NNOut, classNum, layerNum, delta_w_acc)
    % output layer
    [trainNum,~] = size(Y);
    o_k = zeros(classNum, trainNum);
    zip_Y = [1:trainNum; Y']';
    o_k((zip_Y(:,1)-1)*classNum + zip_Y(:,2)) = 1; 
    o_k = o_k';
    
    delta_h{layerNum} = NNOut{layerNum} - o_k;
    
    % hidden & input layer
    for h = layerNum-1: -1: 1
        delta = delta_h{h+1} * NNet(h+1).w(1:end-1, :)';
        delta_h{h} = (NNOut{h} .* (1 - NNOut{h})) .* delta;
    end
    
    % update weights 
    % all delta should be calculated before weights update
    % output & hiden layer
    for i = layerNum: -1: 2
        delta_w_acc{i} = delta_w_acc{i} + [NNOut{i-1} ones(trainNum, 1)]' * delta_h{i};
    end

    % input layer
    delta_w_acc{1} = delta_w_acc{1} + [X ones(trainNum, 1)]' * delta_h{1};
end