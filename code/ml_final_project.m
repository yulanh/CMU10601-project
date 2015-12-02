%% prepare training data
XTrain = [];
YTrain = [];

load ../subset_CIFAR10/small_data_batch_1
XTrain = [XTrain;data];
YTrain = [YTrain;labels];

load ../subset_CIFAR10/small_data_batch_2
XTrain = [XTrain;data];
YTrain = [YTrain;labels];

load ../subset_CIFAR10/small_data_batch_3
XTrain = [XTrain;data];
YTrain = [YTrain;labels];

load ../subset_CIFAR10/small_data_batch_4
XTrain = [XTrain;data];
YTrain = [YTrain;labels];

cellSize = 4;


%% prepare testing data
load ../subset_CIFAR10/small_data_batch_5
XTest = data;
YTest = labels;

Model = train(XTrain, YTrain);
nn_get_acc(Model, XTrain, YTrain)
nn_get_acc(Model, XTest, YTest)
% [XTrainFeat, YTrain] = data2feat(XTrain, YTrain, cellSize);
% [XTestFeat, YTest] = data2feat(XTest, YTest, cellSize);
% 
% %% NN train
% 
% layerNum = 2;
% hiddenNum = 100;
% classNum = 10;
% eta = 0.1;
% maxIter = 32;
% train_data = XTrainFeat;
% train_gold = YTrain;
% test_data = XTestFeat;
% test_gold = YTest;
% 
% NNet = nnTrain(layerNum, hiddenNum, classNum, eta, maxIter, train_data, train_gold, test_data, test_gold);
% save('Model.mat', 'NNet');
% end