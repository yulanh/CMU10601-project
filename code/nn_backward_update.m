function NNet = nn_backward_update(NNet, delta_w_acc, eta, layerNum, batchSize)
    % update weights 
    % all delta should be calculated before weights update
    % output & hiden layer
    for i = layerNum: -1: 2
        NNet(i).w = NNet(i).w - eta * delta_w_acc{i} ;
    end

    % input layer
    NNet(1).w = NNet(1).w - eta * delta_w_acc{1} ;

end