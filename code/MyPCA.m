function [V] = MyPCA(X, n)
    [~, p] = size(X);
    mu = mean(X);
    
    for i = 1 : p
       X( :, i) = X( :, i) - mu(i); 
    end
    sigma = cov(X);
    
    [VOriginal, ~] = eig(sigma);
    V = VOriginal( : , p - n + 1 : p);
end