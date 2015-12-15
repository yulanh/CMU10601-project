function Model = train3(data, labels)
    classNum = 10;
    cellSize = 4;
    C = Inf;
    eps = 1e-12;
    mu = 1e-12;
    
    [XTrain, YTrain] = svm_extract_feat(data, labels, cellSize);
%     save('SVMFeature1000_PCA1000.mat', 'XTrain', 'YTrain');
%     load 'SVMFeature1000_PCA.mat'
    [alpha, b, sv_x, sv_y] = svm_train(XTrain, YTrain, classNum, C, eps, mu);
    Model = struct('alpha', {alpha}, 'b', {b}, 'sv_x', {sv_x}, 'sv_y', {sv_y});
    save('Model_svm.mat', 'Model');
end

function [alpha, b, sv_x, sv_y] = svm_train(X, YClass, classNum, C,  eps, mu)
    
    [trainNum, ~] = size(X);
    alpha = cell(classNum,1);
    b = cell(classNum,1);
    sv_x = cell(classNum,1);
    sv_y = cell(classNum,1);
    
    for class = 1:classNum
        
        Y = zeros(trainNum, 1);
        Y(YClass == class) = 1;
        Y(Y == 0) = -1;
        
        % X -- N*D, Y -- N*1
%         H = diag(Y) * (X * X') * diag(Y);
        H = (Y*Y') .* (X*X') + mu * eye(trainNum);
        
        f = - ones(trainNum, 1);
        
        alp = quadprog(H, f, [], [], Y', 0, zeros(trainNum,1), C*ones(trainNum,1));
        % zeros(trainNum,1), C*ones(trainNum,1) back

        sv_ind = alp>eps;
        bound_ind = alp>eps & alp<(C-eps);
        Xm = X(bound_ind, :);
        Ym = Y(bound_ind);
        Xsv = X(sv_ind, :);
        Ysv = Y(sv_ind);
        alpsv = alp(sv_ind);
        [Nm, ~] = size(Xm);

        b{class} = sum(Ym - sum(repmat(alpsv.*Ysv,[1,Nm]) .* (Xsv*Xm'))') / Nm; 
        
        alpha{class} = alpsv;
        sv_x{class} = Xsv;
        sv_y{class} = Ysv;
    end

end
