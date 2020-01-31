%% High dimensionality dataset
%% Load and split data
format short;
data=zeros(21263,82);
data=readtable('train.csv'); %read the data from excel sheet
data=table2array(data);

% 60% for training, 20% for validation kai 20% testing
training_data=zeros(12757,82);
training_data=data(1 : 12757,:);

evaluation_data=zeros(4252,82);
evaluation_data=data(12758:17010,:);

testing_data=zeros(4252,82);
testing_data=data(17011:21263,:);

%% Normalize data
%normalize data subtracting and dividing with min and max values of
%training data, for having same range
training_data_min= min(training_data(:));
training_data_max=max(training_data(:));
training_data(:) = (training_data(:) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]


evaluation_data_min=min(evaluation_data(:));
evaluation_data_max=max(evaluation_data(:));
evaluation_data(:) = (evaluation_data(:) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]

testing_data_min=min(testing_data(:));
testing_data_max=max(testing_data(:));
testing_data(:) = (testing_data(:) - training_data_min) / (training_data_max - training_data_min); % Scaled to [0, 1]

%% Relief selection of predictors
%with relief algorithm we  find the most important characteristics
% we store the weigts and the indexes 

[idx,weights] =relieff(training_data(:,1:81),training_data(:,82),100);
relief_array=zeros(length(idx),2);
relief_array(:,1)=idx;
 relief_array(:,2)=weights;
[relief_array,index] = sortrows(relief_array,2,'descend');


%% Grid search
%number of rules and features

NR=[3,5,7,9,12];
NF=[3,5,7,10];
errors_array=zeros(length(NF),length(NR));%array for final errors of each rule-feature combination

for i = 1:length(NF)
    for j=1:length(NR)
        
        
     Sets = cvpartition(training_data(:, end), 'KFold', 5);%create test sets
        cvSetError = zeros(Sets.NumTestSets,1);%array of errors for sets of cvpartition

        
         genfis_opt = genfisOptions('FCMClustering','FISType','sugeno');
         genfis_opt.NumClusters = NR(j);%set the number of rules in j iteration
         initial_fis = genfis(training_data(:,relief_array(1:NF(i),1)),training_data(:,82),genfis_opt);%as input we give the features that relief algorithm selected

        %for each set of parameters we have k iterations,one for each set
        %of data partitioning
        for k=1:Sets.NumTestSets
            
            training_set=Sets.training(k);
            testing_set=Sets.test(k);
            
            training_input=training_data(training_set,relief_array(1:NF(i),1));
            training_output=training_data(training_set,82);
            
            testing_input=training_data(testing_set,relief_array(1:NF(i),1));
            testing_output=training_data(testing_set,82);
            
            anfis_opt = anfisOptions('InitialFIS', initial_fis, 'EpochNumber', 40, 'ValidationData', [testing_input testing_output]);
            
            [train_fis,trainError,stepSize,chkFIS,chkError] = anfis([training_input training_output],anfis_opt);
            
            system_output = evalfis(chkFIS,testing_data(:,relief_array(1:NF(i),1)));
            
            cvSetError(k)=sum((system_output - testing_data(:, 82)).^2);
            
        end
        
      model_error=sum(cvSetError)/Sets.NumTestSets;
      errors_array(i,j)=model_error(1)/length(system_output);%store the error at the final error array for all the pairs
    end 
end
%% Plot the results
%we create a bar diagram for each value of the feature parameter,and the
%values of the rules parameter
figure;
subplot(2,2,1);
bar(errors_array(1,:));
title('error with 3 predictors');
xlabel('number of rules');
ylabel('error');
xticklabels({'3','5','7','9','12'})
saveas(gcf, 'error_3_pr.png');
      
subplot(2,2,2);
bar(errors_array(2,:));
title('error with 5 predictors');
xlabel('number of rules');
ylabel('error');
xticklabels({'3','5','7','9','12'})
saveas(gcf, 'error_5_pr.png');

subplot(2,2,3);
bar(errors_array(3,:));
title('error with 7 predictors');
xlabel('number of rules');
ylabel('error');
xticklabels({'3','5','7','9','12'})
saveas(gcf, 'error_7_pr.png');

subplot(2,2,4);
bar(errors_array(4,:));
title('error with 10 predictors');
xlabel('number of rules');
ylabel('error');
xticklabels({'3','5','7','9','12'});
saveas(gcf, 'error_10_pr.png');
