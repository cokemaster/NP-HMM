function [Tr, pI, phi, gamma, xi] = train_hmm_em(O, Nm, K, num_iter, display)
% Input:
%   O - M x T matrix of observations
%   Nm - 1 x M vector of observation variable sizes
%   K - number of hidden states
% Output:
%   Tr - transition matrix [K x K]
%   pi - probabilities of first element [K x 1]
%   gamma - state marginals [K x T]
%   xi - state pair marginals [K x K x T]
%   phi - observation probabilities cell(1, M) of [K x Nm]
    if nargin < 5
        display = false;
    end
    [M, T] = size(O);
    % initialize output variables
    Tr = rand(K);
    Tr = bsxfun(@rdivide, Tr, sum(Tr,2));
    pI = rand(1, K);
    pI = pI/sum(pI);
    phi = cell(1, M);
    for i = 1:M
        phi{i} = ones(K, Nm(i))/prod(Nm);
    end
    gamma = ones(K, T)/K;
    xi = ones(K, K, T)/(K^2);
    % em iterations
    for i = 1:num_iter
        % E - step
        % observation messages
        mu_ot_st = ones(K, T);
        for j = 1:M
            mu_ot_st = mu_ot_st.*phi{j}(:,O(j,:));
        end
        mu_ot_st = bsxfun(@rdivide, mu_ot_st, sum(mu_ot_st));
        % forward
        mu_st_ft = zeros(K, T);
        mu_ftm1_st = zeros(K, T);
        % first state
        mu_ftm1_st(:, 1) = pI;
        for j = 1:(T-1)
            mu_st_ft(:, j) = mu_ot_st(:, j).*mu_ftm1_st(:, j);
            mu_ftm1_st(:, j+1) = (Tr')*mu_st_ft(:, j);
            % normalization
            mu_st_ft(:, j) = mu_st_ft(:, j)/sum(mu_st_ft(:, j));
            mu_ftm1_st(:, j+1) = mu_ftm1_st(:, j+1)/sum(mu_ftm1_st(:, j+1));
        end
        mu_ftm1_st(:, T) = Tr*mu_st_ft(:, T-1);
        % backward
        mu_stp1_ft = zeros(K, T);
        mu_ft_st = zeros(K, T);
        % last state
        mu_ft_st(:, T) = ones(K, 1);
        for j = (T-1):-1:1
            mu_stp1_ft(:, j) = mu_ft_st(:, j+1).*mu_ot_st(:, j+1);
            mu_ft_st(:, j) = Tr*mu_stp1_ft(:, j);
            % normalization
            mu_stp1_ft(:, j) = mu_stp1_ft(:, j)/sum(mu_stp1_ft(:, j));
            mu_ft_st(:, j) = mu_ft_st(:, j)/sum(mu_ft_st(:, j));
        end
        gamma = mu_ftm1_st.*mu_ot_st.*mu_ft_st;
        gamma = bsxfun(@rdivide, gamma, sum(gamma)); % normalization
        xi(:, :, 1) = zeros(K);
        for j = 2:T
            xi(:, :, j) = (mu_ot_st(:, j-1).*mu_ftm1_st(:, j-1))*...
                ((mu_ot_st(:, j).*mu_ft_st(:, j))');
            xi(:, :, j) = xi(:, :, j)/sum(sum(xi(:, :, j))); % normalization
        end
        % M - step
        TrOld = Tr;
        pIOld = pI;
        Tr = bsxfun(@rdivide, sum(xi, 3), sum(gamma(:, 1:T-1), 2));
        pI = gamma(:, 1);
        % Compute log-likelyhood
        marginal_st = mu_ftm1_st.*mu_ftm1_st;
        marginal_st = bsxfun(@rdivide, marginal_st, sum(marginal_st));
        marginal_ot = sum(marginal_st.*mu_ot_st);
        logLl = sum(log(marginal_ot));
        if display
            disp(['iter: ', num2str(i),...
                ' dTr: ', num2str(sqrt(sum((Tr(:) - TrOld(:)).^2))),...
                ' dpI: ', num2str(sqrt(sum((pI(:) - pIOld(:)).^2))),...
                ' logLl: ', num2str(logLl)])
        end
        Z = 0;
        for j = 1:M
            phi{j} = zeros(K, Nm(j));
            for t = 1:Nm(j)
                phi{j}(:, t) = gamma * (O(j,:)' == t);
            end
            %phi{j} = gamma * bsxfun(@eq, O(j,:)', 1:Nm(j));
            Z = Z + sum(phi{j}(:));
        end
        % normalize
        for j = 1:M
            phi{j} = phi{j}/Z;
        end
    end
end