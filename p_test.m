%%%%%%%%% Simple comparison of both algorithms with Wilcoxon Rank Sum test

clear all
clc

SearchAgents_no = 50; % Number of search solutions
Max_iteration = 100;    % Maximum number of iterations

% Pre-allocate arrays for storing results
results_matrix = zeros(30, 2); % 30 functions, 2 algorithms (AOA, MAOA)
p_values = zeros(1, 30);% Stores p-values from Wilcoxon Rank Sum test
summary_matrix = zeros(30, 3);

for i = 1:30 % Loop over each function from F1 to F30
    if i==2
        continue;
    end
    F_name = ['F', num2str(i)]; % Name of the test function
    [lb,ub,dim,fobj]=CEC2017(F_name);
    C3=2;
    C4=0.5;
    
    % Run multiple times and store best scores for each algorithm
    aoa_scores = zeros(1, 30);
    maoa_scores = zeros(1, 30);
    for run = 1:30
        SearchAgents_no=randi([45, 49]);
        [aoa_score, ~ , ~] = GOA2(50, Max_iteration, lb, ub , dim , fobj);
        [maoa_score, ~ , ~] = GOA5(SearchAgents_no, Max_iteration, lb, ub , dim , fobj);
        aoa_scores(run) = aoa_score;
        maoa_scores(run) = maoa_score;
    end
    
    % Calculate Wilcoxon Rank Sum test and store p-value
    [p_values(i), ~, ~] = ranksum(aoa_scores, maoa_scores);

    % Update results matrix with best score for each run
    results_matrix(i, 1) = min(aoa_scores); % Assuming lower score is better
    results_matrix(i, 2) = min(maoa_scores); % Assuming lower score is better


    summary_matrix(i, 1) = mean(aoa_scores);
    summary_matrix(i, 2) = mean(maoa_scores);
    summary_matrix(i, 3) = p_values(i);
    
    % Display results (modify as needed)
    disp(['For function ', F_name, ':']);
    disp(['Best score by AOA (average of 30 runs): ', num2str(mean(aoa_scores))]);
    disp(['Best score by MAOA (average of 30 runs): ', num2str(mean(maoa_scores))]);
    disp(['p-value (Wilcoxon Rank Sum): ', num2str(p_values(i))]);
    disp('\n');
end

% Display or further process the results matrix and p-values
disp('Results Matrix:');
disp(results_matrix);
disp('p-values:');
disp(p_values);