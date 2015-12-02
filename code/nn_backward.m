function [NNet] = nn_backward(NNet, X, Y, NNOut, eta, classNum, layerNum)

    delta_h = cell(layerNum,1);

    % output layer
    o_k = zeros(1, classNum);
    o_k(Y) = 1; 
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
        NNet(i).w = NNet(i).w - eta * [NNOut{i-1} 1]' * delta_h{i};
    end

    % input layer
    NNet(1).w = NNet(1).w - eta * [X 1]' * delta_h{1};

end