train_set = load('TrainingSamplesDCT_subsets_8.mat');
alpha = load('Alpha.mat');
alpha = alpha.alpha;

for strategy_idx = 1:2
    if strategy_idx == 1
        strategy = load('Prior_1.mat');
    elseif strategy_idx == 2
        strategy = load('Prior_2.mat');
    end
    
    for dataset_idx = 1:4
        if dataset_idx == 1
            d1_BG = train_set.D1_BG;
            d1_FG = train_set.D1_FG;
        elseif dataset_idx == 2
            d1_BG = train_set.D2_BG;
            d1_FG = train_set.D2_FG;
        elseif dataset_idx == 3
            d1_BG = train_set.D3_BG;
            d1_FG = train_set.D3_FG;
        elseif dataset_idx == 4
            d1_BG = train_set.D4_BG;
            d1_FG = train_set.D4_FG;
        end
        
        bayes_error = [];
        mle_error = [];
        map_error = [];
        n_FG = size(d1_FG,1);
        n_BG = size(d1_BG,1);

        % Loop for different alpha
        for alpha_idx = 1:size(alpha,2)
            cov_0 = create_prior_cov(alpha(alpha_idx), strategy.W0);
            
            % FG
            d1_FG_cov = cov(d1_FG) * (n_FG-1)/n_FG;
            [mu_1_FG, cov_1_FG, mu_pred_FG, cov_pred_FG] = estimate_bayes_params(d1_FG, d1_FG_cov, cov_0, strategy.mu0_FG, n_FG);
            
            % BG
            d1_BG_cov = cov(d1_BG) * (n_BG-1)/n_BG;
            [mu_1_BG, cov_1_BG, mu_pred_BG, cov_pred_BG] = estimate_bayes_params(d1_BG, d1_BG_cov, cov_0, strategy.mu0_BG, n_BG);

            % Prior
            prior_FG = n_FG / (n_FG + n_BG);
            prior_BG = n_BG / (n_FG + n_BG);

            % Bayes-BDR
            bayes_error = [bayes_error apply_classifier('bayes', mu_pred_FG, cov_pred_FG, mu_pred_BG, cov_pred_BG, alpha_idx, dataset_idx, strategy_idx, prior_FG, prior_BG)];

            % ML-BDR
            mle_error = [mle_error apply_classifier('mle', transpose(mean(d1_FG)), d1_FG_cov, transpose(mean(d1_BG)), d1_BG_cov, alpha_idx, dataset_idx, strategy_idx, prior_FG, prior_BG)];
            
            % MAP-BDR
            map_error = [map_error apply_classifier('map', mu_pred_FG, d1_FG_cov, mu_pred_BG, d1_BG_cov, alpha_idx, dataset_idx, strategy_idx, prior_FG, prior_BG)];
        end
        
        % plot
        figure('visible','off');
        plot(alpha,bayes_error,alpha,mle_error,alpha,map_error);
        legend('Predict','ML','MAP');
        set(gca, 'XScale', 'log');
        title('PoE vs Alpha');
        xlabel('Alpha');
        ylabel('PoE');
        saveas(gcf,['Strategy_' int2str(strategy_idx) '_dataset_' int2str(dataset_idx) '_PoEvsAlpha.png']);
        close(gcf);
    end
end

% Zigzag traversal
function output = zigzag(in)
h = 1;
v = 1;
vmin = 1;
hmin = 1;
vmax = size(in, 1);
hmax = size(in, 2);
i = 1;
output = zeros(1, vmax * hmax);

while ((v <= vmax) && (h <= hmax))
    
    if (mod(h + v, 2) == 0)                 % going up
        if (v == vmin)       
            output(i) = in(v, h);        % if we got to the first line
            if (h == hmax)
	            v = v + 1;
            else
                h = h + 1;
            end
            i = i + 1;
        elseif ((h == hmax) && (v < vmax))   % if we got to the last column
            output(i) = in(v, h);
            v = v + 1;
            i = i + 1;
        elseif ((v > vmin) && (h < hmax))    % all other cases
            output(i) = in(v, h);
            v = v - 1;
            h = h + 1;
            i = i + 1;
    end
        
    else                                    % going down
       if ((v == vmax) && (h <= hmax))       % if we got to the last line
            output(i) = in(v, h);
            h = h + 1;
            i = i + 1;
        
       elseif (h == hmin)                   % if we got to the first column
            output(i) = in(v, h);
            if (v == vmax)
	      h = h + 1;
	    else
              v = v + 1;
            end
            i = i + 1;
       elseif ((v < vmax) && (h > hmin))     % all other cases
            output(i) = in(v, h);
            v = v + 1;
            h = h - 1;
            i = i + 1;
       end
    end
    if ((v == vmax) && (h == hmax))          % bottom right element
        output(i) = in(v, h);
        break
    end
end
end

% Create prior covariance
function cov_0 = create_prior_cov(alpha_val, W0)
    cov_0 = zeros(64,64);
    for idx = 1:64
       cov_0(idx,idx) = alpha_val * W0(idx); 
    end
end

% Estimate Bayesian parameters
function [mu_1, cov_1, mu_pred, cov_pred] = estimate_bayes_params(data, data_cov, cov_0, mu0, n_samples)
    tmp = inv(cov_0 + (1/n_samples)*data_cov);
    mu_1 = cov_0 * tmp * transpose(mean(data)) + (1/n_samples) * data_cov * tmp * transpose(mu0);
    cov_1 = cov_0 * tmp * (1/n_samples) * data_cov;
    % predictive distribution (normal distribution)
    mu_pred = mu_1;
    cov_pred = data_cov + cov_1;
end

% Apply classifier
function total_error = apply_classifier(classifier_type, mu_FG, cov_FG, mu_BG, cov_BG, alpha_idx, dataset_idx, strategy_idx, prior_FG, prior_BG)
    % Load and process image
    img = imread('cheetah.bmp');
    img = im2double(img);
    % Add paddle
    img = [zeros(size(img,1),2) img];
    img = [zeros(2, size(img,2)); img];
    img = [img zeros(size(img,1),5)];
    img = [img; zeros(5, size(img,2))];

    [m,n] = size(img);
    Blocks = ones(m-7,n-7);
    
    % Prepare constants for discriminant function
    det_cov_FG = det(cov_FG);
    det_cov_BG = det(cov_BG);
    ave_tmp_FG = transpose(mu_FG);
    ave_tmp_BG = transpose(mu_BG);
    inv_tmp_FG = inv(cov_FG);
    inv_tmp_BG = inv(cov_BG);

    % Pre-compute constants
    const_FG = ave_tmp_FG*inv_tmp_FG*transpose(ave_tmp_FG) + log(det_cov_FG) - 2*log(prior_FG);
    const_BG = ave_tmp_BG*inv_tmp_BG*transpose(ave_tmp_BG) + log(det_cov_BG) - 2*log(prior_BG);
    
    % Classify
    for i=1:m-7
        for j=1:n-7
            DCT = dct2(img(i:i+7,j:j+7));
            zigzag_order = zigzag(DCT);
            feature = zigzag_order;
            
            g_cheetah = compute_score(feature, inv_tmp_FG, ave_tmp_FG, const_FG);
            g_grass = compute_score(feature, inv_tmp_BG, ave_tmp_BG, const_BG);
            
            if g_cheetah >= g_grass
                Blocks(i,j) = 0;
            end
        end
    end

    % Save prediction
    imwrite(Blocks, [classifier_type '_prediction_alpha_' int2str(alpha_idx) '_dataset_' int2str(dataset_idx) '_strategy_' int2str(strategy_idx) '.jpg']);
    prediction = mat2gray(Blocks);

    % Calculate error
    total_error = calculate_error(prediction, prior_FG, prior_BG);
end

% Compute discriminant score
function score = compute_score(feature, inv_cov, ave_tmp, const)
    score = feature*inv_cov*transpose(feature) - 2*feature*inv_cov*transpose(ave_tmp) + const;
end

% Calculate classification error
function total_error_64 = calculate_error(prediction, prior_FG, prior_BG)
    ground_truth = imread('cheetah_mask.bmp')/255;
    x = size(ground_truth, 1);
    y = size(ground_truth, 2);
    count1 = 0;
    count2 = 0;
    count_cheetah_truth = 0;
    count_grass_truth = 0;
    
    for i=1:x
        for j=1:y
            if prediction(i,j) > ground_truth(i,j)
                count2 = count2 + 1;
                count_grass_truth = count_grass_truth + 1;
            elseif prediction(i,j) < ground_truth(i,j)
                count1 = count1 + 1;
                count_cheetah_truth = count_cheetah_truth + 1;
            elseif ground_truth(i,j) >0
                count_cheetah_truth = count_cheetah_truth + 1;
            else
                count_grass_truth = count_grass_truth + 1;
            end
        end
    end
    
    error1_64 = (count1/count_cheetah_truth) * prior_FG;
    error2_64 = (count2/count_grass_truth) * prior_BG;
    total_error_64 = error1_64 + error2_64;
end