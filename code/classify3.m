function Y = classify3(Model, data)
    classNum = 10;
    cellSize = 4;
    
    b = Model.b;
    alpha = Model.alpha;
    sv_x = Model.sv_x;
    sv_y = Model.sv_y;
    
    [X, ~] = svm_extract_feat(data, [], cellSize);
    [dataNum, ~] = size(X);
    
    svm_out = zeros(classNum, dataNum);
    for class = 1:classNum
        c_b = b{class};
        c_alpha = alpha{class};
        c_sv_x = sv_x{class};
        c_sv_y = sv_y{class};

        svm_out(class,:) = sum(repmat(c_alpha.*c_sv_y, [1,dataNum]) .* (c_sv_x*X')) + c_b;

    end
    
    [~, pred_y] = max(svm_out);
    Y = pred_y';
    
% two class
%     Y = -ones(dataNum,1);
%     Y(svm_out>0) = 1;
    
end