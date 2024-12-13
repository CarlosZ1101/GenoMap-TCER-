%%
clear
% Add necessary code folders
addpath(genpath('gromov-wassersteinOT'))

% Load data
load TMdata
data_new=double(data);
% Row and column number of Genomap images
rowN=33;
colN=33;

genomaps=construct_genomap(data_new,rowN,colN);

%% data
load label_TMData
load index_TM

GT=grp2idx(label_TMData);
GTX=categorical(GT);
dataMat_CNNtrain=genomaps(:,:,:,indxTrain);
dataMat_CNNtest=genomaps(:,:,:,indxTest);
groundTruthTest=GTX(indxTest);
groundTruthTrain=GTX(indxTrain);

label_TMDataU=unique(label_TMData);

[GTsorted,idxGT]=sort(groundTruthTrain);

dataSorted_train=dataMat_CNNtrain(:,:,:,idxGT);

[GTsorted_test,idxGT_test]=sort(groundTruthTest);

dataSorted_test=dataMat_CNNtest(:,:,:,idxGT_test);

% Save the sorted training data to a file
save('dataSorted_train.mat', 'dataSorted_train');

% Save the sorted test data to a file
save('dataSorted_test.mat', 'dataSorted_test');

save('dataMat_CNNtest.mat','dataMat_CNNtest');

save('dataMat_CNNtrain.mat','dataMat_CNNtrain');


newName_dataSorted_train = 'train_genoMaps_GT';
newName_dataSorted_test = 'test_genoMaps_GT';
newName_dataMat_CNNtest = 'test_genoMaps';
newName_dataMat_CNNtrain = 'train_genoMaps';

train_genoMaps_GT=dataSorted_train;
test_genoMaps_GT = dataSorted_test;
test_genoMaps = dataMat_CNNtest;
train_genoMaps = dataMat_CNNtrain;

save('CellularTax_dataSAVER10-2000.mat', 'train_genoMaps_GT', 'test_genoMaps_GT', 'test_genoMaps', 'train_genoMaps');

save('CellularTax_GTlabel','GTlabel')
