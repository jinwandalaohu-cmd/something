%% SVR模型训练与优化 - 水中油浓度预测（仅梦境优化，无多余代码）
clear; clc; close all;

%% 1. 数据加载
train_file = '训练集样本.xlsx';
test_file  = '测试集样本.xlsx';

fprintf('正在加载训练数据...\n');
try
    [train_data, ~, ~] = xlsread(train_file, 'Sheet1');
    non_nan_rows = find(~all(isnan(train_data(:, 1:5)), 2));
    if isempty(non_nan_rows), error('训练数据中没有找到有效数据'); end
    start_row = non_nan_rows(1);
    end_row   = non_nan_rows(end);
    y_train = train_data(start_row:end_row, 1);
    X_train = train_data(start_row:end_row, 2:5);
    fprintf('训练数据加载成功！样本数: %d\n', size(X_train,1));
catch
    try
        train_table = readtable(train_file, 'Sheet', 'Sheet1');
        numeric_data = table2array(train_table(:, 1:5));
        valid_rows = ~any(isnan(numeric_data),2);
        numeric_data = numeric_data(valid_rows,:);
        y_train = numeric_data(:,1);
        X_train = numeric_data(:,2:5);
        fprintf('备用方法加载成功，样本数: %d\n', size(X_train,1));
    catch ME
        error('训练数据加载失败: %s', ME.message);
    end
end

fprintf('正在加载测试数据...\n');
try
    X_test1 = xlsread(test_file, 'Sheet1', 'B3:E7');
    y_test1 = xlsread(test_file, 'Sheet1', 'A3:A7');
    X_test2 = xlsread(test_file, 'Sheet1', 'B12:E16');
    y_test2 = xlsread(test_file, 'Sheet1', 'A12:A16');
    X_test3 = xlsread(test_file, 'Sheet1', 'B21:E25');
    y_test3 = xlsread(test_file, 'Sheet1', 'A21:A25');
    fprintf('测试数据加载成功！\n');
catch
    test_data = xlsread(test_file, 'Sheet1');
    X_test1 = test_data(3:7, 2:5);   y_test1 = test_data(3:7, 1);
    X_test2 = test_data(12:16,2:5);  y_test2 = test_data(12:16,1);
    X_test3 = test_data(21:25,2:5);  y_test3 = test_data(21:25,1);
    fprintf('备用方法加载测试数据成功！\n');
end

fprintf('训练集: %d 样本, 测试集1: %d, 测试集2: %d, 测试集3: %d\n', ...
    size(X_train,1), size(X_test1,1), size(X_test2,1), size(X_test3,1));

%% 2. 数据预处理
fprintf('\n=== 数据预处理 ===\n');
[X_train_scaled, mu_X, sigma_X] = zscore(X_train);
[y_train_scaled, mu_y, sigma_y] = zscore(y_train);
X_test1_scaled = (X_test1 - mu_X) ./ sigma_X;
X_test2_scaled = (X_test2 - mu_X) ./ sigma_X;
X_test3_scaled = (X_test3 - mu_X) ./ sigma_X;
fprintf('标准化完成。\n');

%% 3. 定义参数范围
param_range = struct();
param_range.C       = logspace(-2, 3, 50);
param_range.epsilon = logspace(-4, -1, 50);
param_range.gamma   = logspace(-4, 1, 50);

%% 4. 超强梦境优化算法
fprintf('\n=== 开始超强梦境优化 ===\n');
tic;
n_iterations = 100;
n_dreams = 15;
temperature = 1.0;
cooling_rate = 0.95;

% 初始化梦境
dreams = cell(n_dreams,1);
dream_scores = inf(n_dreams,1);
best_score = inf;
best_params = struct();

fprintf('初始化梦境...\n');
for i = 1:n_dreams
    dreams{i} = randomSampleParamsSafe(param_range);
    dream_scores(i) = evaluateSVR(dreams{i}, X_train_scaled, y_train_scaled);
    if dream_scores(i) < best_score
        best_score = dream_scores(i);
        best_params = dreams{i};
    end
    if mod(i,5)==0
        fprintf('  梦境 %d/%d, 当前最佳 R² = %.4f\n', i, n_dreams, -best_score);
    end
end

% 主循环
for iter = 1:n_iterations
    temperature = temperature * cooling_rate;
    for i = 1:n_dreams
        current_params = dreams{i};
        current_score = dream_scores(i);
        new_params = generateNewParams(current_params, best_params, param_range, temperature, iter, n_iterations);
        if ~validateParams(new_params), continue; end
        try
            new_score = evaluateSVR(new_params, X_train_scaled, y_train_scaled);
        catch
            new_score = current_score + 10;
        end
        if new_score < current_score
            dreams{i} = new_params;
            dream_scores(i) = new_score;
        else
            delta = new_score - current_score;
            if rand() < exp(-delta / temperature)
                dreams{i} = new_params;
                dream_scores(i) = new_score;
            end
        end
        if new_score < best_score
            best_score = new_score;
            best_params = new_params;
        end
    end
    if mod(iter,10)==0 || iter==1 || iter==n_iterations
        fprintf('  迭代 %d/%d, 温度=%.3f, 最佳 R² = %.6f\n', iter, n_iterations, temperature, -best_score);
    end
end
time_dream = toc;
fprintf('\n梦境优化完成！最佳参数: C=%.6f, ε=%.8f, γ=%.8f\n', ...
    best_params.C, best_params.epsilon, best_params.gamma);
fprintf('最佳 R² = %.6f, 耗时 = %.2f 秒\n', -best_score, time_dream);

%% 5. 使用最佳参数训练 SVR 并预测三个测试集
fprintf('\n=== 训练最终模型并预测 ===\n');
model = fitrsvm(X_train_scaled, y_train_scaled, ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', max(best_params.C, 1e-6), ...
    'Epsilon', max(best_params.epsilon, 1e-8), ...
    'KernelScale', 1/sqrt(max(best_params.gamma, 1e-8)), ...
    'Standardize', false);

y_pred1 = predict(model, X_test1_scaled) * sigma_y + mu_y;
y_pred2 = predict(model, X_test2_scaled) * sigma_y + mu_y;
y_pred3 = predict(model, X_test3_scaled) * sigma_y + mu_y;

[rmse1, mae1, r2_1, mape1] = calculateMetrics(y_test1, y_pred1);
[rmse2, mae2, r2_2, mape2] = calculateMetrics(y_test2, y_pred2);
[rmse3, mae3, r2_3, mape3] = calculateMetrics(y_test3, y_pred3);

fprintf('测试集1: RMSE=%.4f, MAE=%.4f, R²=%.4f, MAPE=%.2f%%\n', rmse1, mae1, r2_1, mape1);
fprintf('测试集2: RMSE=%.4f, MAE=%.4f, R²=%.4f, MAPE=%.2f%%\n', rmse2, mae2, r2_2, mape2);
fprintf('测试集3: RMSE=%.4f, MAE=%.4f, R²=%.4f, MAPE=%.2f%%\n', rmse3, mae3, r2_3, mape3);

fprintf('\n=== 程序运行结束 ===\n');

%% ========================== 辅助函数 ==========================
function score = evaluateSVR(params, X, y)
    if ~validateParams(params)
        score = 100; return;
    end
    n_folds = min(5, size(X,1));
    cv = cvpartition(length(y), 'KFold', n_folds);
    cv_scores = zeros(n_folds,1);
    for fold = 1:n_folds
        X_tr = X(training(cv,fold), :);
        y_tr = y(training(cv,fold));
        X_te = X(test(cv,fold), :);
        y_te = y(test(cv,fold));
        model = fitrsvm(X_tr, y_tr, 'KernelFunction','rbf', ...
            'BoxConstraint', max(params.C,1e-6), ...
            'Epsilon', max(params.epsilon,1e-8), ...
            'KernelScale', 1/sqrt(max(params.gamma,1e-8)), ...
            'Standardize',false);
        y_pred = predict(model, X_te);
        ss_res = sum((y_te - y_pred).^2);
        ss_tot = sum((y_te - mean(y_te)).^2);
        if ss_tot == 0, r2 = 1; else, r2 = 1 - ss_res/ss_tot; end
        cv_scores(fold) = r2;
    end
    score = -mean(cv_scores);
end

function params = randomSampleParamsSafe(pr)
    params.C       = 10^(log10(max(pr.C(1),1e-6))       + rand()*(log10(pr.C(end))       - log10(max(pr.C(1),1e-6))));
    params.epsilon = 10^(log10(max(pr.epsilon(1),1e-8)) + rand()*(log10(pr.epsilon(end)) - log10(max(pr.epsilon(1),1e-8))));
    params.gamma   = 10^(log10(max(pr.gamma(1),1e-8))   + rand()*(log10(pr.gamma(end))   - log10(max(pr.gamma(1),1e-8))));
end

function new = perturbParamsSafe(p, scale, pr)
    new.C       = max(min(p.C       * (1 + scale*(2*rand()-1)), pr.C(end)),       max(pr.C(1),1e-6));
    new.epsilon = max(min(p.epsilon * (1 + scale*(2*rand()-1)), pr.epsilon(end)), max(pr.epsilon(1),1e-8));
    new.gamma   = max(min(p.gamma   * (1 + scale*(2*rand()-1)), pr.gamma(end)),   max(pr.gamma(1),1e-8));
end

function new = generateNewParams(cur, best, pr, temp, iter, maxIter)
    explore_prob = 0.3 * (1 - iter/maxIter);
    if rand() < explore_prob
        scale = 0.5 * temp;
        new = perturbParamsSafe(cur, scale, pr);
    else
        lr = 0.3;
        new.C       = cur.C       * (1-lr) + best.C       * lr;
        new.epsilon = cur.epsilon * (1-lr) + best.epsilon * lr;
        new.gamma   = cur.gamma   * (1-lr) + best.gamma   * lr;
        fine_scale = 0.1 * temp;
        new.C       = max(min(new.C       * (1 + fine_scale*(2*rand()-1)), pr.C(end)),       max(pr.C(1),1e-6));
        new.epsilon = max(min(new.epsilon * (1 + fine_scale*(2*rand()-1)), pr.epsilon(end)), max(pr.epsilon(1),1e-8));
        new.gamma   = max(min(new.gamma   * (1 + fine_scale*(2*rand()-1)), pr.gamma(end)),   max(pr.gamma(1),1e-8));
    end
end

function ok = validateParams(p)
    ok = isstruct(p) && all(isfield(p,{'C','epsilon','gamma'})) && ...
         p.C>0 && p.epsilon>0 && p.gamma>0 && ...
         all(isfinite([p.C, p.epsilon, p.gamma]));
end

function [rmse, mae, r2, mape] = calculateMetrics(trueVal, predVal)
    valid = ~isnan(trueVal) & ~isnan(predVal);
    trueVal = trueVal(valid);
    predVal = predVal(valid);
    if isempty(trueVal)
        rmse=NaN; mae=NaN; r2=NaN; mape=NaN; return;
    end
    rmse = sqrt(mean((trueVal - predVal).^2));
    mae  = mean(abs(trueVal - predVal));
    ss_res = sum((trueVal - predVal).^2);
    ss_tot = sum((trueVal - mean(trueVal)).^2);
    r2 = 1 - ss_res/ss_tot;
    idx = trueVal ~= 0;
    if any(idx)
        mape = mean(abs((trueVal(idx)-predVal(idx))./trueVal(idx))) * 100;
    else
        mape = 0;
    end
end