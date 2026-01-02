function main()
    config = setup_configuration();
    [train_FG, train_BG] = load_training_data('TrainingSamplesDCT_8_new.mat');
    [img, ground_truth] = load_test_data('cheetah.bmp', 'cheetah_mask.bmp');
    results = run_experiments(train_FG, train_BG, img, ground_truth, config);
    visualize_results(results, config);
end

function config = setup_configuration()
    config.num_comp = 8;  
    config.dim_list = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64];
    config.num_mixtures = 5;
    config.max_iterations = 200;
    config.prior_FG = 250 / (250 + 1053);
    config.prior_BG = 1053 / (250 + 1053);
end

function [train_FG, train_BG] = load_training_data(filename)
    data = load(filename);
    train_FG = data.TrainsampleDCT_FG;
    train_BG = data.TrainsampleDCT_BG;
end

function [img, ground_truth] = load_test_data(img_file, mask_file)
    img = imread(img_file);
    img = im2double(img);
    img = pad_image(img, 7);
    ground_truth = imread(mask_file) / 255;
end

function padded_img = pad_image(img, pad_size)
    [rows, cols] = size(img);
    padded_img = zeros(rows + pad_size, cols + pad_size);
    padded_img(1:rows, 1:cols) = img;
end

function results = run_experiments(train_FG, train_BG, img, ground_truth, config)
    fprintf('\n=== Training Phase ===\n');
    
    fprintf('Training FG mixtures...\n');
    models_FG = cell(config.num_mixtures, 1);
    for i = 1:config.num_mixtures
        fprintf('  Training FG mixture %d/%d...\n', i, config.num_mixtures);
        models_FG{i} = train_single_mixture(train_FG, config.num_comp, config, 'FG', i);
    end
    
    fprintf('Training BG mixtures...\n');
    models_BG = cell(config.num_mixtures, 1);
    for i = 1:config.num_mixtures
        fprintf('  Training BG mixture %d/%d...\n', i, config.num_mixtures);
        models_BG{i} = train_single_mixture(train_BG, config.num_comp, config, 'BG', i);
    end
    
    fprintf('\n Evaluation Phase \n');
    results.error_matrix = cell(config.num_mixtures, config.num_mixtures);
    results.dimensions = config.dim_list;
    results.num_mixtures = config.num_mixtures;
    for fg_idx = 1:config.num_mixtures
        fprintf('Evaluating FG mixture %d with all BG mixtures \n', fg_idx);
        for bg_idx = 1:config.num_mixtures
            fprintf('  FG-%d with BG-%d: ', fg_idx, bg_idx);
            error_rates = evaluate_mixture_pair(img, ground_truth, ...
                models_FG{fg_idx}, models_BG{bg_idx}, config);
            results.error_matrix{fg_idx, bg_idx} = error_rates;
            
            fprintf('Done (min=%.4f, max=%.4f)\n', min(error_rates), max(error_rates));
        end
    end
end

function gmm = train_single_mixture(train_data, num_comp, config, class_name, mix_idx)
    rng(mix_idx * 1000 + sum(class_name));
    init_params = get_init_params(class_name);
    gmm = initialize_gmm(num_comp, init_params);
    gmm = em_algorithm(train_data, gmm, config.max_iterations);
end

function params = get_init_params(class_name)
    if strcmp(class_name, 'FG')
        params.mean_val = 5;
        params.std_val = 0.3;
        params.offset = 1;
    else
        params.mean_val = 5;
        params.std_val = 0.1;
        params.offset = 5;
    end
end

function gmm = initialize_gmm(num_comp, params)
    feature_dim = 64;
    
    gmm.num_comp = num_comp;
    gmm.mu = zeros(num_comp, feature_dim);
    gmm.cov = cell(num_comp, 1);
    gmm.pi = ones(1, num_comp) / num_comp;
    
    for c = 1:num_comp
        cov_vals = abs(normrnd(params.mean_val, params.std_val, [1, feature_dim])) + params.offset;
        cov_vals = cov_vals / sum(cov_vals);
        cov_vals = cov_vals + 1e-6;
        gmm.cov{c} = diag(cov_vals);
        mu_vals = normrnd(params.mean_val, params.std_val, [1, feature_dim]);
        gmm.mu(c, :) = mu_vals / sum(mu_vals);
    end
end

function gmm = em_algorithm(train_data, gmm, max_iterations)
    for iter = 1:max_iterations
        responsibilities = compute_responsibilities(train_data, gmm);
        gmm = update_gmm_parameters(train_data, gmm, responsibilities);
    end
end

function responsibilities = compute_responsibilities(data, gmm)
    n_samples = size(data, 1);
    n_comp = gmm.num_comp;
    
    likelihoods = zeros(n_samples, n_comp);
    
    for c = 1:n_comp
        try
            cov_c = gmm.cov{c};
            if any(diag(cov_c) <= 0)
                cov_c = cov_c + eye(size(cov_c)) * 1e-6;
                gmm.cov{c} = cov_c;
            end
            pdf_vals = mvnpdf(data, gmm.mu(c, :), cov_c);
            pdf_vals(pdf_vals < 1e-300) = 1e-300;
            pdf_vals(isnan(pdf_vals)) = 1e-300;
            pdf_vals(isinf(pdf_vals)) = 1e-300;
            
            likelihoods(:, c) = gmm.pi(c) * pdf_vals;
            
        catch ME
            warning('Component %d: %s. Using fallback likelihood.', c, ME.message);
            likelihoods(:, c) = gmm.pi(c) * 1e-300;
        end
    end
    total_likelihood = sum(likelihoods, 2);
    total_likelihood(total_likelihood < 1e-300) = 1e-300;
    responsibilities = likelihoods ./ total_likelihood;
    responsibilities(isnan(responsibilities)) = 1 / n_comp;
    responsibilities(isinf(responsibilities)) = 1 / n_comp;
    row_sums = sum(responsibilities, 2);
    responsibilities = responsibilities ./ row_sums;
end

function gmm = update_gmm_parameters(data, gmm, responsibilities)
    n_samples = size(data, 1);
    n_comp = gmm.num_comp;
    reg_term = 1e-6;
    
    for c = 1:n_comp
        n_c = sum(responsibilities(:, c));
        n_c = max(n_c, 1e-10);
        gmm.pi(c) = (n_c + reg_term) / (n_samples + n_comp * reg_term);
        gmm.mu(c, :) = sum(responsibilities(:, c) .* data, 1) / n_c;
        diff = data - gmm.mu(c, :);
        weighted_sq_diff = sum(responsibilities(:, c) .* (diff .^ 2), 1);
        cov_diag = weighted_sq_diff / n_c + reg_term;
        cov_diag = max(cov_diag, reg_term);
        gmm.cov{c} = diag(cov_diag);
    end
    gmm.pi = gmm.pi / sum(gmm.pi);
end

function error_rates = evaluate_mixture_pair(img, ground_truth, gmm_FG, gmm_BG, config)
    [rows, cols] = size(img);
    num_dims = length(config.dim_list);
    error_rates = zeros(1, num_dims);
    
    for dim_idx = 1:num_dims
        dim = config.dim_list(dim_idx);
        predictions = generate_predictions(img, gmm_FG, gmm_BG, dim, config, rows, cols);
        error_rates(dim_idx) = calculate_error_rate(predictions, ground_truth, config);
    end
end

function predictions = generate_predictions(img, gmm_FG, gmm_BG, dim, config, rows, cols)
    predictions = zeros(rows - 7, cols - 7);
    
    for i = 1:(rows - 7)
        for j = 1:(cols - 7)
            block = img(i:i+7, j:j+7);
            dct_block = dct2(block);
            features = zigzag_scan(dct_block);
            features = features(1:dim);
            predictions(i, j) = classify_pixel(features, gmm_FG, gmm_BG, dim, config);
        end
    end
end

function prediction = classify_pixel(features, gmm_FG, gmm_BG, dim, config)
    likelihood_FG = compute_mixture_likelihood(features, gmm_FG, dim);
    likelihood_BG = compute_mixture_likelihood(features, gmm_BG, dim);
    posterior_FG = likelihood_FG * config.prior_FG;
    posterior_BG = likelihood_BG * config.prior_BG;
    
    prediction = double(posterior_FG >= posterior_BG);
end

function likelihood = compute_mixture_likelihood(features, gmm, dim)
    likelihood = 0;
    
    for c = 1:gmm.num_comp
        mu_c = gmm.mu(c, 1:dim);
        cov_c = gmm.cov{c}(1:dim, 1:dim);
        min_eig = min(eig(cov_c));
        if min_eig <= 0
            cov_c = cov_c + eye(dim) * (abs(min_eig) + 1e-6);
        end
        
        try
            component_likelihood = gmm.pi(c) * mvnpdf(features, mu_c, cov_c);
            if isnan(component_likelihood) || isinf(component_likelihood)
                component_likelihood = 1e-300;
            end
            likelihood = likelihood + component_likelihood;
        catch
            continue;
        end
    end
    
    likelihood = max(likelihood, 1e-300);
end

function error_rate = calculate_error_rate(predictions, ground_truth, config)
    [rows, cols] = size(predictions);
    false_negative = 0;
    false_positive = 0;
    total_cheetah = 0;
    total_grass = 0;
    
    for i = 1:rows
        for j = 1:cols
            pred = predictions(i, j);
            truth = ground_truth(i, j);
            
            if truth > 0
                total_cheetah = total_cheetah + 1;
                if pred == 0
                    false_negative = false_negative + 1;
                end
            else
                total_grass = total_grass + 1;
                if pred > 0
                    false_positive = false_positive + 1;
                end
            end
        end
    end
    
    error_FG = (false_negative / total_cheetah) * config.prior_FG;
    error_BG = (false_positive / total_grass) * config.prior_BG;
    error_rate = error_FG + error_BG;
end

function result = zigzag_scan(matrix)
    zigzag_pattern = [
         1,  2,  6,  7, 15, 16, 28, 29;
         3,  5,  8, 14, 17, 27, 30, 43;
         4,  9, 13, 18, 26, 31, 42, 44;
        10, 12, 19, 25, 32, 41, 45, 54;
        11, 20, 24, 33, 40, 46, 53, 55;
        21, 23, 34, 39, 47, 52, 56, 61;
        22, 35, 38, 48, 51, 57, 60, 62;
        36, 37, 49, 50, 58, 59, 63, 64
    ];
    
    result = zeros(1, 64);
    for i = 1:8
        for j = 1:8
            result(zigzag_pattern(i, j)) = matrix(i, j);
        end
    end
end

function visualize_results(results, config)
    fprintf('\n Generating Plots \n');
    if ~exist('plots', 'dir')
        mkdir('plots');
    end
    fprintf('Verifying results data...\n');
    for fg = 1:config.num_mixtures
        for bg = 1:config.num_mixtures
            data = results.error_matrix{fg, bg};
            if isempty(data)
                error('Missing data for FG-%d, BG-%d', fg, bg);
            end
            fprintf('  FG-%d, BG-%d: %d values (range: %.4f to %.4f)\n', ...
                fg, bg, length(data), min(data), max(data));
        end
    end
    fprintf('\nCreating plots...\n');
    for fg_idx = 1:config.num_mixtures
        create_fg_plot(results, config, fg_idx);
    end
    
    fprintf('All plots saved successfully!\n');
end

function create_fg_plot(results, config, fg_idx)
    fprintf('  Creating plot for FG-%d...\n', fg_idx);
    fig = figure('Position', [100, 100, 1000, 700]);

    colors = [
        0.0000, 0.4470, 0.7410;  % Blue
        0.8500, 0.3250, 0.0980;  % Orange
        0.9290, 0.6940, 0.1250;  % Yellow
        0.4940, 0.1840, 0.5560;  % Purple
        0.4660, 0.6740, 0.1880   % Green
    ];
    markers = {'o', 's', 'd', '^', 'v'};
    
    hold on;

    for bg_idx = 1:config.num_mixtures
        error_data = results.error_matrix{fg_idx, bg_idx};
        
        if length(error_data) ~= length(config.dim_list)
            error('Data length mismatch for FG-%d, BG-%d', fg_idx, bg_idx);
        end
       
        h = plot(config.dim_list, error_data, ...
            'LineWidth', 2, ...
            'Color', colors(bg_idx, :), ...
            'Marker', markers{bg_idx}, ...
            'MarkerSize', 8, ...
            'MarkerFaceColor', colors(bg_idx, :), ...
            'DisplayName', sprintf('BG-%d', bg_idx));
        
        fprintf('    Plotted BG-%d (errors: %.4f to %.4f)\n', ...
            bg_idx, min(error_data), max(error_data));
    end
    
    hold off;
    
    xlabel('dimension', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('P_E', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('FG-%d', fg_idx), 'FontSize', 16, 'FontWeight', 'bold');
    
    legend('Location', 'northeast', 'FontSize', 12);
    
    grid on;
    box on;
    set(gca, 'FontSize', 12);
    
    xlim([0, 65]);
    
    saveas(fig, sprintf('plots/FG_%d.png', fg_idx));
    saveas(fig, sprintf('plots/FG_%d.fig', fg_idx));
    
    fprintf('    Saved FG-%d plot\n', fg_idx);
end