function error = get_acc(Neural_Net,input,expected_out)

    real_out = nn_forward(Neural_Net,input);

    % the result is the node with max value in the real output
    [~,res] = max(real_out{end},[],2);
    error = length(find(res == expected_out)) / length(expected_out);
end
