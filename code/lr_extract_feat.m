function [XFeat, Y] = lr_extract_feat(data, labels, cellSize)
    XFeat = [];

    for i = 1:size(data,1)
        im = im2single(reshape(data(i,:), 32, 32, 3));
        hog = vl_hog(im, cellSize);
        XFeat = [XFeat;hog(:)'];
    end
    
    XFeat = double(XFeat); 
    
    % include bias term in the feature
    % but will be handled differently in train
    XFeat = [ones(size(XFeat,1),1) XFeat];
    
    Y = double(labels) + 1;
end