function [XFeat, Y] = svm_extract_feat(data, labels, cellSize)
    XFeat = [];
%     XFeat = data;
    for i = 1:size(data,1)
        im = im2single(reshape(data(i,:), 32, 32, 3));
        hog = vl_hog(im, cellSize);
        XFeat = [XFeat;hog(:)'];
    end
    
    XFeat = double(XFeat);   
    V = MyPCA(XFeat, 1000);
    XFeat = XFeat * V;
    
    Y = double(labels) + 1;
end