function [XFeat, Y] = nn_extract_feat(data, labels, cellSize)
    XFeat = [];
    for i = 1:size(data,1)
        im = im2single(reshape(data(i,:), 32, 32, 3));
        hog = vl_hog(im, cellSize);
        XFeat = [XFeat;hog(:)'];
    end
    
    
    XFeat = double(XFeat);
    Y = double(labels) + 1;
end