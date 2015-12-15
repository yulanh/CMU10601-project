function error = svm_get_acc(Model,input,expected_out)
    
    res = classify3(Model, input);
    error = length(find(res == expected_out)) / length(expected_out);
end