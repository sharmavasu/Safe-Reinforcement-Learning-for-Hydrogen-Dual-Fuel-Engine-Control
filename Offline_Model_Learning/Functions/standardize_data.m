function [s_norm, mu, sig] = standardize_data(s)
mu = mean(s);
sig = std(s);

s_norm = (s - mu) ./ sig;

% mu = min(data);
% sig = max(data);
% data_n = ((data - mu) / (sig-mu));
end

