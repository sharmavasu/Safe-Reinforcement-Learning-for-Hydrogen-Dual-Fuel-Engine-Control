
%% Construct C2C NMPC

% parameter_file:
% C2C_NMPC.expdata = '1123_002_GRU_normalized';
% load('Par_1123002_0008_0008_0247');

C2C_NMPC.expdata = '1123_002_GRU_standardized';
load('Par_1123002_0008_0008_0271');

% C2C_NMPC.expdata = '1123_002_GRU';
% load('Par_1123002_0008_0008_0245');

H2DF.Par = Par; 
H2DF_Par = H2DF.Par;

% initialization:
C2C_NMPC = init_C2C_NMPC_H2DF(C2C_NMPC.expdata);
C2C_NMPC.Par = Par;
C2C_NMPC.name = C2C_NMPC.name;
C2C_NMPC.Opts.N =2;
C2C_NMPC.Opts.nlp_solver = 'sqp'; % {'sqp', 'sqp_rti'}
C2C_NMPC.Opts.max_sqp_iter = 3;
C2C_NMPC.Opts.steplength = 1;
C2C_NMPC.Opts.gnsf_detect_struct = 'true';
C2C_NMPC.Opts.tol = 1e-3;

C2C_NMPC.x0 = [zeros(C2C_NMPC.Dims.n_LSTM_states, 1); C2C_NMPC.Initialization; zeros(C2C_NMPC.Dims.n_outputs, 1)];
C2C_NMPC.xu0 = C2C_NMPC.Initialization;
[C2C_NMPC.outputs_ref, C2C_NMPC.controls_ref] = generate_reference_h2dual(C2C_NMPC.Dims.n_controls);
C2C_NMPC.controls_ref_norm = normalize_var(C2C_NMPC.controls_ref, C2C_NMPC.Normalization.controls.mean, C2C_NMPC.Normalization.controls.std, 'to-scaled'); 
C2C_NMPC.outputs_ref_norm = normalize_var(C2C_NMPC.outputs_ref, C2C_NMPC.Normalization.outputs.mean, C2C_NMPC.Normalization.outputs.std, 'to-scaled');
C2C_NMPC.ref =[C2C_NMPC.controls_ref;C2C_NMPC.outputs_ref;zeros(C2C_NMPC.Dims.n_controls, length(C2C_NMPC.controls_ref(1, :)))];

Ts = 0.08;%1
Tf = 50;
UpperLimit= ([0.5e-3;900;2;5.5e-3] - C2C_NMPC.Normalization.controls.mean)./(C2C_NMPC.Normalization.controls.std);
LowerLimit = ([0.17e-3;350;-5;1e-3] - C2C_NMPC.Normalization.controls.mean)./(C2C_NMPC.Normalization.controls.std);