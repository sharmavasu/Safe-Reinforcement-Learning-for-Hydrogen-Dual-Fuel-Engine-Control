function [nextState, output] = MyGRUstateFnc_withBatchnorm(state, action, Par)

%
%   Signature   : nextState = dyn_lstm(state, action, Par)
% 
%   Inputs      : state -> State, consisting of concatenation of cell states
%                          and hidden states
%                 action -> Input vector of LSTM
%                 Params -> Struct containing all required weights and
%                           biases
% 
%   Outputs     : nextState -> Updated cell and hidden states
%               : output -> output of system
% 
%-------------------------------------------------------------------------%

% Get cell and hidden states
% cellState = state(1:Par.nCellStates);
hiddenState = state(Par.nCellStates + 1:end);
hiddenState = state;

%------------------------------FC-----------------------------------------%

% Layer 1 - Fully connected
ZFc1 = Par.WFc1 * action + Par.bFc1;
ZFc1 = ReLu_function(ZFc1);
ZFc1 = (ZFc1 - Par.bn_mu1)./sqrt(Par.bn_var1 + Par.bn_epsilon1);

% Layer 2 - Fully connected
ZFc2 = Par.WFc2 * ZFc1 + Par.bFc2;
ZFc2 = ReLu_function(ZFc2);
ZFc2 = (ZFc2 - Par.bn_mu2)./sqrt(Par.bn_var2 + Par.bn_epsilon2);

% Layer 3 - Fully connected
ZFc3 = Par.WFc3 * ZFc2 + Par.bFc3;
ZFc3 = ReLu_function(ZFc3);
ZFc3 = (ZFc3 - Par.bn_mu3)./sqrt(Par.bn_var3 + Par.bn_epsilon3);

% %---------------------------LSTM------------------------------------------%
% % LSTM Layer - Input gate
% it = logistic_function(Par.wi * ZFc3 + Par.Ri * hiddenState + Par.bi);
% 
% % LSTM Layer - Forget gate
% ft = logistic_function(Par.wf * ZFc3 + Par.Rf * hiddenState + Par.bf);
% 
% % LSTM Layer - External input gate (cell candidate)
% gt = tanh(Par.wg * ZFc3 + Par.Rg * hiddenState + Par.bg);
% 
% % LSTM Layer - Update cell states 
% cellStateNext = ft .* cellState + gt .* it;
% 
% % LSTM Layer - Output gate
% ot = logistic_function(Par.wo * ZFc3 + Par.Ro * hiddenState + Par.bo);
% 
% % LSTM Layer - Update hidden states 
% hiddenStateNext = tanh(cellStateNext) .* ot;

%---------------------------GRU------------------------------------------%
z = logistic_function(Par.wz * ZFc3 + Par.Rz * hiddenState + Par.bz);

r = logistic_function(Par.wr * ZFc3 + Par.Rr * hiddenState + Par.br);

h_ = tanh (Par.wh * ZFc3+ r.*Par.Rh* hiddenState+ Par.bh);

hiddenStateNext = z.*hiddenState + (1-z).*h_;


%------------------------------FC-----------------------------------------%
% Layer 4 - Fully connected
ZFc4 = Par.WFc4 * hiddenStateNext + Par.bFc4;
ZFc4 = ReLu_function(ZFc4);
ZFc4 = (ZFc4 - Par.bn_mu4)./sqrt(Par.bn_var4 + Par.bn_epsilon4);

% Layer 4 - Fully connected
ZFc5 = Par.WFc5 * ZFc4 + Par.bFc5;
ZFc5 = ReLu_function(ZFc5);
ZFc5 = (ZFc5 - Par.bn_mu5)./sqrt(Par.bn_var5 + Par.bn_epsilon5);

% Layer 6 - Fully connected
ZFc6 = Par.WFc6 * ZFc5 + Par.bFc6;

%nextState = [cellStateNext; hiddenStateNext];
nextState = hiddenStateNext;
output = ZFc6;

end

%% Auxiliary functions
function y = ReLu_function(x)
y = max(0,x);
end

function y = logistic_function(x)
y = 1 ./ (1 + exp(-x));
end