%%YPATIA DAMI 

format short ;
%% LOAD AND SPLIT DATA
%1:T , 2:AP, 3:RH, 4:V, 5:EO

data=zeros(9570,5);
data=readtable('dataset.xlsx');
data=table2array(data);

% 60% for training, 20% for validation kai 20% testing
training_data=zeros(5742,5);
training_data=data(1 : 5742,:);

evaluation_data=zeros(1914,5);
evaluation_data=data(5743:7654,:);

testing_data=zeros(1915,5);
testing_data=data(7655:9568,:);
%% NORMALIZE DATA

training_data_min= min(training_data(:));
training_data_max=max(training_data(:));
training_data(:) = (training_data(:) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]


evaluation_data_min=min(evaluation_data(:));
evaluation_data_max=max(evaluation_data(:));
evaluation_data(:) = (evaluation_data(:) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]

testing_data_min=min(testing_data(:));
testing_data_max=max(testing_data(:));
testing_data(:) = (testing_data(:) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]

%% TRAIN THE MODEL 
%TSK_model_1 
%number of input mfs : 2
%output : singleton

fis_opt = genfisOptions('GridPartition');
fis_opt.NumMembershipFunctions = [2 2 2 2]; % Two mf for each input variable
fis_opt.OutputMembershipFunctionType = 'constant';
fis_opt.InputMembershipFunctionType = ["gbellmf" "gbellmf" "gbellmf" "gbellmf"]; % Bell-shaped mfs

initial_fis = genfis(training_data(:,1:4),training_data(:,5),fis_opt);

%plot initial member functions
figure;
[x,mf] = plotmf(initial_fis,'input',1);
subplot(2,2,1);
plot(x,mf);
title('input1 - Temperature');
ylabel('Membership function');
saveas(gcf,'tsk_1_input1_initMF.png');

[x,mf] = plotmf(initial_fis,'input',2);
subplot(2,2,2);
plot(x,mf);
title('input2 - Ambient pressure');
ylabel('Membership function');
saveas(gcf,'tsk_1_input2_initMF.png');

[x,mf] = plotmf(initial_fis,'input',3);
subplot(2,2,3);
plot(x,mf);
title('input3 - Relative humidity');
ylabel('Membership function');
saveas(gcf,'tsk_1_input3_initMF.png');

[x,mf] = plotmf(initial_fis,'input',4);
subplot(2,2,4);
plot(x,mf);
title('input4 - Exhaust vacuum');
ylabel('Membership function');
saveas(gcf,'tsk_1_input4_initMF.png');

anfis_opt=anfisOptions('InitialFIS',initial_fis);
anfis_opt.ValidationData=evaluation_data;
anfis_opt.EpochNumber=400;
[train_fis,trainError,stepSize,chkFIS,chkError] = anfis(training_data,anfis_opt);


system_output = evalfis(chkFIS,testing_data(:,1:4));

%plotting final mfs to compare with initial

figure;
[x,mf] = plotmf(chkFIS,'input',1);
subplot(2,2,1);
plot(x,mf);
title('input1 - Temperature');
ylabel('Membership function');
saveas(gcf,'tsk_1_input1_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',2);
subplot(2,2,2);
plot(x,mf);
title('input2 - Ambient pressure');
ylabel('Membership function');
saveas(gcf,'tsk_1_input2_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',3);
subplot(2,2,3);
plot(x,mf);
title('input3 - Relative humidity');
ylabel('Membership function');
saveas(gcf,'tsk_1_input3_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',4);
subplot(2,2,4);
plot(x,mf);
title('input4 - Exhaust vacuum');
ylabel('Membership function');
saveas(gcf,'tsk_1_input4_finalMF.png');

%% MODEL EVALUATION

final_error=system_output-testing_data(:,5);   

mse=  sum(final_error.^2)/length(final_error);

rmse=sqrt(mse);

SSres=sum((testing_data(:,5)-system_output).^2);

SStot=sum((testing_data(:,5)-(sum(testing_data(:,5)/length(testing_data(:,5)))).^2));

R2=1-SSres/SStot;

nmse = mse/var(testing_data(:,5));

ndei=sqrt(nmse);

fprintf('mse = %f rmse = %f R2 = %f nmse = %f ndei = %f\n', mse, rmse, R2, nmse, ndei);

%% PLOTTING RESULTS

%Prediction error of fuzzy system
figure;
plot(1:length(final_error),final_error,'b');
title('Prediction Error');
xlabel('Iterations');
ylabel('PredictionError');
saveas(gcf,'PredictionError1.png');

%fuzzy system results
figure;
plot(1:length(system_output),system_output,'r',1:length(system_output),testing_data(1:length(system_output),5),'b');
title('Fuzzy System output');
legend('Fuzzy System Output','Testing data Output');
saveas(gcf,'FuzzyOutput1.png');

%training and evaluating errors
figure;
plot(1:length(trainError),trainError,1:length(trainError),chkError);
title('Learning Curves');
legend('Training Error', 'Testing Error');
saveas(gcf,'TrainingErrors1.png');

%mse = 0.000017 rmse = 0.004150 R2 = 0.999930 nmse = 0.063336 ndei = 0.251667
