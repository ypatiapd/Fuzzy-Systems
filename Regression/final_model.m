%%YPATIA DAMI 
%% High dimensionality dataset-Final model
%% Load and split data


data=zeros(21263,82);
data1=readtable('train.csv');
data=table2array(data1);

shuffled_data = zeros(size(data));
rand_pos = randperm(length(data)); %array of random positions
% new array with original data randomly distributed
for k = 1:length(data)
    shuffled_data(k, :) = data(rand_pos(k), :);
end

%% Relief selection of predictors
%efarmozoume ton algorithmo tou relief gia tin epilogi xaraktiristikwn

[idx,weights] =relieff(shuffled_data(:,1:81),shuffled_data(:,82),100);
relief_array=zeros(length(idx),2);
relief_array(:,1)=idx;
 relief_array(:,2)=weights;
[relief_array,index] = sortrows(relief_array,2,'descend');

genfis_opt.NumClusters = 7;
numofPred=10;

% 60% for training, 20% for validation kai 20% testing
training_data=zeros(12757,82);
training_data=shuffled_data(1 : 12757,:);

evaluation_data=zeros(4252,82);
evaluation_data=shuffled_data(12758:17010,:);

testing_data=zeros(4252,82);
testing_data=shuffled_data(17011:21263,:);

%% Normalize data
%logw diaforetikou eurous timwn kanonikopoioume kathe xaraktiristiko xwrista
%vriskoume diladi topiko elaxisto kai megisto 
%kanonikopoioume sto diastima [0,1]

 for i = 1 : size(training_data,2)
    training_data_min = min(training_data(:,i));
    training_data_max = max(training_data(:,i));
    training_data(:,i) = (training_data(:,i) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]
   
    evaluation_data(:,i) = (evaluation_data(:,i) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]

    testing_data(:,i) = (testing_data(:,i) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]
end
%% chosen predictors

input_training_data=training_data(:,relief_array(1:numofPred,1));
output_training_data=training_data(:,82);

input_evaluation_data=evaluation_data(:,relief_array(1:numofPred,1));
output_evaluation_data=evaluation_data(:,82);

input_testing_data=testing_data(:,relief_array(1:numofPred,1));
output_testing_data=testing_data(:,82);

%% TRAIN THE MODEL 
%FINAL MODEL 
%number of predictors : 10
%number of rules : 5

 genfis_opt = genfisOptions('FCMClustering','FISType','sugeno');
 
 genfis_opt.NumClusters=5;

initial_fis = genfis(training_data(:,relief_array(1:numofPred,1)),training_data(:,82),genfis_opt);

%plot initial member functions
figure;
[x,mf] = plotmf(initial_fis,'input',1);
subplot(2,2,1);
plot(x,mf);

title('input1 ');
ylabel('Membership function');
saveas(gcf,'final_model_input1_initMF.png');

[x,mf] = plotmf(initial_fis,'input',2);
subplot(2,2,2);
plot(x,mf);
title('input2 ');
ylabel('Membership function');
saveas(gcf,'final_model_input2_initMF.png');

[x,mf] = plotmf(initial_fis,'input',3);
subplot(2,2,3);
plot(x,mf);
title('input3');
ylabel('Membership function');
saveas(gcf,'final_model_input3_initMF.png');

[x,mf] = plotmf(initial_fis,'input',4);
subplot(2,2,4);
plot(x,mf);
title('input4');
ylabel('Membership function');
saveas(gcf,'final_model_input4_initMF.png');


anfis_opt=anfisOptions('InitialFIS',initial_fis);
anfis_opt.ValidationData=[input_evaluation_data output_evaluation_data];
anfis_opt.EpochNumber=200;
[train_fis,trainError,stepSize,chkFIS,chkError] = anfis([input_training_data output_training_data],anfis_opt);


system_output = evalfis(input_testing_data,chkFIS);

%plotting final mfs to compare with initial

figure;
[x,mf] = plotmf(chkFIS,'input',1);
subplot(2,2,1);
plot(x,mf);
title('input1');
ylabel('Membership function');
saveas(gcf,'final_model_input1_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',2);
subplot(2,2,2);
plot(x,mf);
title('input2 ');
ylabel('Membership function');
saveas(gcf,'final_model_input2_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',3);
subplot(2,2,3);
plot(x,mf);
title('input3 ');
ylabel('Membership function');
saveas(gcf,'final_model_input3_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',4);
subplot(2,2,4);
plot(x,mf);
title('input3 ');
ylabel('Membership function');
saveas(gcf,'final_model_input3_finalMF.png');


%% MODEL EVALUATION

final_error=system_output-output_testing_data;   

mse=  sum(final_error.^2)/length(final_error);

rmse=sqrt(mse);

SSres=sum((output_testing_data-system_output).^2);

SStot=sum((output_testing_data-(sum(output_testing_data/length(output_testing_data))).^2));

R2=1-SSres/SStot;

nmse = mse/var(output_testing_data);

ndei=sqrt(nmse);

fprintf('mse = %f rmse = %f R2 = %f nmse = %f ndei = %f\n', mse, rmse, R2, nmse, ndei);

%% PLOTTING RESULTS



%fuzzy system results

figure;

plot(1:length(system_output),system_output,'r*',1:length(system_output),testing_data(1:length(system_output),82),'b*');
title('Fuzzy System output');
legend('Fuzzy System Output','Testing data Output');
saveas(gcf,'FuzzyOutput_finalmodel.png');

%training and evaluating errors

figure;

plot(1:length(trainError),trainError,1:length(trainError),chkError);
title('Learning Curves');
legend('Training Error', 'Testing Error');
saveas(gcf,'TrainingErrors_finalmodel.png');


