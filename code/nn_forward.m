function NNOut = nn_forward(NNet, X)

    l_input = [X ones(size(X, 1), 1)];

    NNOut = cell(length(NNet), 1);
    for l = 1: length(NNet)
        wx = l_input * NNet(l).w;

        if strcmp(NNet(l).f, 'sigmoid')
            NNOut{l} = 1 ./ (1 + exp(-wx));

        elseif strcmp(NNet(l).f, 'softmax')
            
            K = size(wx, 2);
            for k = 1:K
                NNOut{l}(:, k) = 1 ./ sum(exp(wx - repmat(wx(:, k),[1 K])), 2);
            end
            
        else
            
            error('Unknown function');
        end

        % input for next layer is the output for the current layer
        l_input = [NNOut{l} ones(size(NNOut{l}, 1), 1)];
    end
end


