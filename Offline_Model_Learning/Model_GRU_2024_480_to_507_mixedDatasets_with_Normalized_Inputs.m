%{ 
Authors:    Armin Norouzi(arminnorouzi2016@gmail.com),            
            David Gordon(dgordon@ualberta.ca),
            Eugen Nuss(e.nuss@irt.rwth-aachen.de)
            Alexander Winkler(winkler_a@mmp.rwth-aachen.de)
            Vasu Shamra(vasu3@ualberta.ca),


Copyright 2023 MECE,University of Alberta,
               Teaching and Research 
               Area Mechatronics in Mobile Propulsion,
               RWTH Aachen University

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
%}

clear
close all
clc
%% Set PWD & Path
% Windows Alex
% cd('C:/GIT/h2df_model_training/H2DFmodel/Scripts') % change directory to where the scripts are 
% addpath('../Functions/') %for mygrustateFnc etc.
% addpath('C:/GIT/matlab2tikz/src') %for tikz (pull repo online by yourself: https://github.com/matlab2tikz/matlab2tikz)

% Linux UofA AI machine Alex
cd ('C:\Users\vasu3\Documents\Work\h2df_model_training\H2DFmodel\Scripts') % change directory to where the scripts are 
addpath('..\Functions\') %for mygrustateFnc etc.
addpath('C:\Users\vasu3\Documents\Work\matlab2tikz\src') %for tikz (pull repo online by yourself: https://github.com/matlab2tikz/matlab2tikz)

%% Settings
do_training = false; 
break_loop = true;
plot_vars = true; 
plot_vars_datasets = true;
plot_explainability = true;
plot_pred_val = true;
plot_pred_test = true;
plot_test_small = false;
plot_val_small = true;
plot_init = true; % plotting to investigate dataset
plot_init_measurements = true; % look on lvl 1 and lvl2 counters of the individual measurements
kill_violated_data = true; % kill lvl1 and lvl2 hits datapoints
kill_points_zero_h2_doi = true; % when safety lvl 2 was hit, standard controlelr was on until the end of the feq cycle, so kill these cycles here
positive_IMEP = true;
generate_tikz = true; 
verify_my_func = false;


Opts.fontsize =20;

ratio_train = 0.80;
ratio_val = 0.95;

MP = 2024;
trainingrun = 271; % start, will be increased in case of grid search
% 252: without lvl 1 hit data 0.0066
% 253: with lvl 1 hit dta, 80% data split 0.0055
% 254: with lvl 1 hit dta, 80% data split, no feedback IMEP 0.059
% 255: with lvl 1 hit dta, 80% data split, no feedback IMEP 
% 255: with lvl 1 hit dta, 80% data split, no feedback IMEP, run #2
% 256: with lvl 1 hit dta, 80% data split, no feedback IMEP, run #2,
% 0.005787
% 257: with lvl 1 hit dta, 80% data split, no feedback IMEP, run #2,
% 0.005787, numhidde units 6
% 262: with lvl 1 hit dta, 80% data split (15% val, 5% test)
% 26X: with lvl 1 hit dta, 80% data split (15% val, 5% test)
% 263: best of the 262, 263, 264 batch
% 265: see below, got overwritten
% 266-267: added test dataset of 5% and rmse. new structure to script. rmse
% constant, but better docs and approach like this. 265 got overwritten.
% 265: first shot with additional data, now 100k points
% 268: additonal data, now 100k points, o,o0516 
% 269-XXX (grid search): new run without the  h2doi=0 points, where standard controller was
% working

%% Load data
% call function
RNG = true;
load('Test_480_to_481_conc.mat')
data_conc.fpga_lastCycle_MPRR = 10*data_conc.fpga_lastCycle_MPRR; % dp max, in Mpa per CAD
[utrain_4801, ytrain_4801, uval_4801, yval_4801, utest_4801, ytest_4801] = getDatasets(data_conc, 1, 0 ,ratio_train, ratio_val, plot_init_measurements, kill_violated_data, kill_points_zero_h2_doi,positive_IMEP,RNG);
% function [utrain, ytrain, uval, yval, utest, ytest] = getDatasets(data_conc, idx_start, ratio_train, ratio_val, plot_init, kill_violated_data)

load('Test_502_NoPress.mat')
[utrain_502, ytrain_502, uval_502, yval_502, utest_502, ytest_502] = getDatasets(Test_502_NoPress, 1, 0, ratio_train, ratio_val, plot_init_measurements, kill_violated_data, kill_points_zero_h2_doi,positive_IMEP,RNG);

load('Test_507_NoPress.mat')
[utrain_507, ytrain_507, uval_507, yval_507, utest_507, ytest_507] = getDatasets(Test_507_NoPress, 1, 0, ratio_train, ratio_val, plot_init_measurements, kill_violated_data, kill_points_zero_h2_doi,positive_IMEP,RNG);

savename_data = '2024_480_to_507_GRU_standardized_post.mat';
savename_datasets = '2024_480_to_507_GRU_standardized_datasets.mat';
savename_datasets_phys = '2024_480_to_507_GRU_phys_datasets.mat';

%concatenate
utrain = [utrain_4801'; utrain_502'; utrain_507'];
ytrain = [ytrain_4801'; ytrain_502'; ytrain_507'];

uval = [uval_4801'; uval_502'; uval_507'];
yval = [yval_4801'; yval_502'; yval_507'];

utest = [utest_4801'; utest_502'; utest_507'];
ytest = [ytest_4801'; ytest_502'; ytest_507'];

utotal = [utrain; uval; utest];
ytotal = [ytrain; yval; ytest];

% get dat from concatenated datasets
DOI_main_cycle = utotal(:,1)';
P2M_cycle = utotal(:,2)';
SOI_main_cycle = utotal(:,3)';
H2_doi_cycle = utotal(:,4)'; % convert s to ms
IMEP_old = utotal(:,5)';
IMEP_cycle = ytotal(:,1)'; % pressure, bascically load in MPa
NOx_cycle = ytotal(:,2)'; % cheap CAN sensor, not FTIR! in ppm
Soot_cycle = ytotal(:,3)'; % in mgm3
MPRR_cycle = ytotal(:,4)'; % dp max, in Mpa per CAD

% Concatenate all signals into one matrix
disp('Minimum and Maximum Values:');
disp(['DOI_main_cycle: Min = ', num2str(min(DOI_main_cycle)), ', Max = ', num2str(max(DOI_main_cycle))]);
disp(['P2M_cycle: Min = ', num2str(min(P2M_cycle)), ', Max = ', num2str(max(P2M_cycle))]);
disp(['SOI_main_cycle: Min = ', num2str(min(SOI_main_cycle)), ', Max = ', num2str(max(SOI_main_cycle))]);
disp(['H2_doi_cycle: Min = ', num2str(min(H2_doi_cycle)), ', Max = ', num2str(max(H2_doi_cycle))]);
disp(['IMEP_old: Min = ', num2str(min(IMEP_old)), ', Max = ', num2str(max(IMEP_old))]);
disp(['IMEP_cycle: Min = ', num2str(min(IMEP_cycle)), ', Max = ', num2str(max(IMEP_cycle))]);
disp(['NOx_cycle: Min = ', num2str(min(NOx_cycle)), ', Max = ', num2str(max(NOx_cycle))]);
disp(['Soot_cycle: Min = ', num2str(min(Soot_cycle)), ', Max = ', num2str(max(Soot_cycle))]);
disp(['MPRR_cycle: Min = ', num2str(min(MPRR_cycle)), ', Max = ', num2str(max(MPRR_cycle))]);
%% mkdir
mkdir('../Plots/'+ sprintf("%04d",MP)',sprintf('%04d',trainingrun))
mkdir('../','/Results/')

max_scale = length(utotal); % 65024 for VSR1123002

%% if plot_vars (plot inputs, outputs, histogram whole dataset)
if plot_vars
%% ploting inputs whole dataset
f = figure;
set(gcf, 'Position', [100, 100, 1800, 1000]);
set(gcf,'color','w');

ax5=subplot(4,1,1);
plot(H2_doi_cycle, 'k','LineWidth',1.5)
grid on
xlabel("\#Cycles",'Interpreter', 'latex')
ylabel({'H2. DOI', '[s]'},'Interpreter','latex')
xlim([0,max_scale])
set(gca,'FontSize',10)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

ax6=subplot(4,1,2);
plot(DOI_main_cycle, 'k','LineWidth',2)
grid on
xlabel("\#Cycles",'Interpreter', 'latex')
ylabel({'Main Inj. DOI', '[s]'},'Interpreter','latex')
xlim([0,max_scale])
set(gca,'FontSize',10)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

ax7=subplot(4,1,3);
plot(P2M_cycle, 'k','LineWidth',1.5)
grid on
xlabel("\#Cycles",'Interpreter', 'latex')
ylabel({'P2M', '[us]'},'Interpreter','latex')
xlim([0,max_scale])
set(gca,'FontSize',10)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

ax8=subplot(4,1,4);
plot(SOI_main_cycle, 'k','LineWidth',1.5)
grid on
xlabel("\#Cycles",'Interpreter', 'latex')
ylabel({'SOI Main','[bTDC CAD]'},'Interpreter','latex')
xlim([0,max_scale])
set(gca,'FontSize',10)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

set(gcf,'units','points','position',[200,200,900,400])

if generate_tikz
    figFileName="../Plots/"+ sprintf("%04d",MP)+"/Inputs";
    savefig(figFileName);
    saveas(gcf,figFileName,"jpg");
    cleanfigure('targetResolution', 200)
    matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false);
    %export_fig(figFileName,'-eps');
    exportgraphics(f,strcat(figFileName, '.pdf'),'BackgroundColor','none','ContentType','vector')
end

%% ploting outputs whole dataset
f = figure;
set(gcf, 'Position', [100, 100, 1800, 800]);
set(gcf,'color','w');

%--------------------------------------------------
ax1=subplot(4,1,1);
plot(IMEP_cycle, 'k','LineWidth',1.5)
grid on
xlabel("\#Cycles",'Interpreter', 'latex')
ylabel({'IMEP',' [Pa]'},'Interpreter','latex')
xlim([0,max_scale])
set(gca,'FontSize',10)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
%--------------------------------------------------
ax2=subplot(4,1,2);
plot(NOx_cycle, 'k','LineWidth',1.5)
grid on
xlabel("\#Cycles",'Interpreter', 'latex')
ylabel({'NO$_x$', '[ppm]'},'Interpreter','latex')
xlim([0,max_scale])
set(gca,'FontSize',10)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
%--------------------------------------------------
ax3=subplot(4,1,3);
plot(Soot_cycle, 'k','LineWidth',1.5)
grid on
xlabel("\#Cycles",'Interpreter', 'latex')
ylabel({'Soot',' [mg/m$^3$]'},'Interpreter','latex')
xlim([0,max_scale])
set(gca,'FontSize',10)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
%--------------------------------------------------
ax4=subplot(4,1,4);
plot(MPRR_cycle, 'k','LineWidth',1.5)
grid on
xlabel("\#Cycles",'Interpreter', 'latex')
ylabel({'MPRR',' [Pa/ 0.1 CA $^\circ$]'},'Interpreter','latex')
xlim([0,max_scale])
set(gca,'FontSize',10)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

linkaxes([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8],'x');

set(gcf,'units','points','position',[200,200,900,400])

if generate_tikz
    figFileName="../Plots/"+ sprintf("%04d",MP)+"/Outputs";
    savefig(figFileName);
    saveas(gcf,figFileName,"jpg");
    cleanfigure('targetResolution', 20)
    matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false);
    %export_fig(figFileName,'-eps');
    exportgraphics(f,strcat(figFileName, '.pdf'),'BackgroundColor','none','ContentType','vector')
end

%% Histogram on whole dataset
Opts.fontsize = 38;
Opts.axissize = 33;  % Define axis label size

% IMEP Data Distribution
f = figure;
set(gcf, 'color', 'w');
set(gcf, 'Position', [100, 100, 1200, 800]);
histogram(IMEP_cycle, NumBins=30);
set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
grid on;
xlabel({'IMEP / Pa'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
ylabel("\#Data Points / -", 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
%title('IMEP Data Distribution', 'Interpreter', 'latex', 'FontSize', Opts.fontsize, 'FontWeight', 'bold');

if generate_tikz
    figFileName = "../Plots/" + sprintf("%04d", MP) + "/IMEP_Data_Distribution";
    savefig(figFileName);
    saveas(gcf, figFileName, "jpg");
    cleanfigure('targetResolution', 20);
    matlab2tikz(convertStringsToChars(figFileName + '.tex'), 'showInfo', false);
    exportgraphics(f, strcat(figFileName, '.pdf'), 'BackgroundColor', 'none', 'ContentType', 'vector');
end

% NOx Data Distribution
f = figure;
set(gcf, 'color', 'w');
set(gcf, 'Position', [100, 100, 1200, 800]);
histogram(NOx_cycle, NumBins=30);
set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
grid on;
xlabel({'NOx / ppm'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
ylabel("\#Data Points / -", 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
%title('NOX Data Distribution', 'Interpreter', 'latex', 'FontSize', Opts.fontsize, 'FontWeight', 'bold');

if generate_tikz
    figFileName = "../Plots/" + sprintf("%04d", MP) + "/NOX_Data_Distribution";
    savefig(figFileName);
    saveas(gcf, figFileName, "jpg");
    cleanfigure('targetResolution', 20);
    matlab2tikz(convertStringsToChars(figFileName + '.tex'), 'showInfo', false);
    exportgraphics(f, strcat(figFileName, '.pdf'), 'BackgroundColor', 'none', 'ContentType', 'vector');
end

% MPRR Data Distribution
f = figure;
set(gcf, 'color', 'w');
set(gcf, 'Position', [100, 100, 1200, 800]);
histogram(MPRR_cycle, NumBins=30);
set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
grid on;
xlabel({'MPRR / Pa/ 0.1 CA $^\circ$'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
ylabel("\#Data Points / -", 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
%title('MPRR Data Distribution', 'Interpreter', 'latex', 'FontSize', Opts.fontsize, 'FontWeight', 'bold');

if generate_tikz
    figFileName = "../Plots/" + sprintf("%04d", MP) + "/MPRR_Data_Distribution";
    savefig(figFileName);
    saveas(gcf, figFileName, "jpg");
    cleanfigure('targetResolution', 20);
    matlab2tikz(convertStringsToChars(figFileName + '.tex'), 'showInfo', false);
    exportgraphics(f, strcat(figFileName, '.pdf'), 'BackgroundColor', 'none', 'ContentType', 'vector');
end

% Soot Data Distribution
f = figure;
set(gcf, 'color', 'w');
set(gcf, 'Position', [100, 100, 1200, 800]);
histogram(Soot_cycle, NumBins=30);
set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
grid on;
xlabel("Soot / mg/m$^3$", 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
ylabel("\#Data Points / -", 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
%title('Soot Data Distribution', 'Interpreter', 'latex', 'FontSize', Opts.fontsize, 'FontWeight', 'bold');

if generate_tikz
    figFileName = "../Plots/" + sprintf("%04d", MP) + "/SOOT_Data_Distribution";
    savefig(figFileName);
    saveas(gcf, figFileName, "jpg");
    cleanfigure('targetResolution', 20);
    matlab2tikz(convertStringsToChars(figFileName + '.tex'), 'showInfo', false);
    exportgraphics(f, strcat(figFileName, '.pdf'), 'BackgroundColor', 'none', 'ContentType', 'vector');
end

%% ploting inputs histogram whole dataset
f =figure;
set(gcf, 'Position', [100, 100, 1800, 1000]);
set(gcf,'color','w');

ax51=subplot(4,1,1);
histogram(IMEP_cycle)
grid on
xlabel({'IMEP [Pa]'},'Interpreter','latex')
ylabel("\#Data Points",'Interpreter', 'latex')
title('IMEP Data Distribution','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])

ax61=subplot(4,1,2);
histogram(NOx_cycle) 
grid on
xlabel({'NOx [ppm]'},'Interpreter','latex')
ylabel("\#Data Points",'Interpreter', 'latex')
title('NOX Data Distribution','Interpreter', 'latex')

ax71=subplot(4,1,3);
set(gcf,'color','w');
histogram(Soot_cycle)
grid on
ylabel({'\#Data Points'},'Interpreter','latex')
xlabel("Soot [mg/m$^3$]",'Interpreter', 'latex')
title('Soot Data Distribution','Interpreter', 'latex')

ax81=subplot(4,1,4);
histogram(MPRR_cycle)
grid on
xlabel({'MPRR [Pa/ 0.1 CA $^\circ$]'},'Interpreter','latex')
ylabel("\#Data Points",'Interpreter', 'latex')
title('MPRR Data Distribution','Interpreter', 'latex')

% set(gcf,'units','points','position',[200,200,900,400])

if generate_tikz
    figFileName="../Plots/"+ sprintf("%04d",MP)+"/Input_Distribution";
    savefig(figFileName);
    saveas(gcf,figFileName,"jpg");
    cleanfigure('targetResolution', 400)
    matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false);
    %export_fig(figFileName,'-eps');
    exportgraphics(f,strcat(figFileName, '.pdf'),'BackgroundColor','none','ContentType','vector')

end
end

%% analysis array init
runs_total_max = 30;
analysis = struct();
analysis.FinalRMSE = zeros(runs_total_max, 1);
analysis.FinalValidationLoss = zeros(runs_total_max, 1);
analysis.TotalLearnables = zeros(runs_total_max, 1);
analysis.ElapsedTime = zeros(runs_total_max, 1);
A = ['XXXX_00YY_00YY_0ZZZ.mat'];
analysis.savename = repmat(A, runs_total_max, 1);
run_nmbr = 0;


%% plot histograms and steps on the different datasets (train, val, test)
if plot_vars_datasets == true 

IMEP_cycle_tr = ytrain(:,1);
IMEP_cycle_val = yval(:,1);
IMEP_cycle_test = ytest(:,1);
NOx_cycle_tr = ytrain(:,2);
NOx_cycle_val = yval(:,2);
NOx_cycle_test = ytest(:,2);
Soot_cycle_tr = ytrain(:,3);
Soot_cycle_val = yval(:,3);
Soot_cycle_test = ytest(:,3);
MPRR_cycle_tr = ytrain(:,4);
MPRR_cycle_val = yval(:,4);
MPRR_cycle_test = ytest(:,4);
max_scale = max( [max(ytrain(:,1)), max(yval(:,1)), max(ytest(:,1))]);

Opts.fontsize = 18;
Opts.axissize = 14; % Define axis label size

f = figure;
set(gcf,'color','w');
set(gcf,'units','points','position',[200,200,900,400])

subplot(3,1,1)
histogram(IMEP_cycle_tr, NumBins=30)
grid on
xlim([0,max_scale])
title('Training Dataset','Interpreter', 'latex')
set(gca,'FontSize',Opts.axissize,'TickLabelInterpreter','latex')
ax = gca;

subplot(3,1,2)
histogram(IMEP_cycle_val, NumBins=30)
grid on
xlim([0,max_scale])
title('Validation Dataset','Interpreter', 'latex')
set(gca,'FontSize',Opts.axissize,'TickLabelInterpreter','latex')
ylabel("\#Data Points / -",'Interpreter', 'latex', 'FontSize', Opts.fontsize)
ax1 = gca;

subplot(3,1,3)
histogram(IMEP_cycle_test, NumBins=30)
grid on
xlim([0,max_scale])
title('Test Dataset','Interpreter', 'latex')
set(gca,'FontSize',Opts.axissize,'TickLabelInterpreter','latex')
xlabel({'IMEP / Pa'},'Interpreter','latex', 'FontSize', Opts.fontsize)
ax2 = gca;

linkaxes([ax ax1 ax2],'x')

if generate_tikz
 figFileName="../Plots/"+ sprintf("%04d",MP)+"/IMEP_Data_Distribution_combined";
 savefig(figFileName);
 saveas(gcf,figFileName,"jpg");
 cleanfigure('targetResolution', 20)
 matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false);
 exportgraphics(f,strcat(figFileName, '.pdf'),'BackgroundColor','none','ContentType','vector')
end

% % NOx
% figure
% set(gcf,'color','w');
% histogram(NOx_cycle) 
% grid on
% xlabel({'NOx [ppm]'},'Interpreter','latex')
% ylabel("\#Data Points",'Interpreter', 'latex')
% title('NOX Training Data Distribution','Interpreter', 'latex')
% 
% if generate_tikz
%     figFileName="../Plots/"+ sprintf("%04d",MP)+"/NOX_Data_Distribution";
%     savefig(figFileName);
%     saveas(gcf,figFileName,"jpg");
%     cleanfigure('targetResolution', 20)
%     matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false);
%     %export_fig(figFileName,'-eps');
% end
% 
% % MPRR
% figure
% set(gcf,'color','w');
% histogram(MPRR_cycle)
% grid on
% xlabel({'MPRR [Pa]'},'Interpreter','latex')
% ylabel("\#Data Points",'Interpreter', 'latex')
% title('MPRR Training Data Distribution','Interpreter', 'latex')
% 
% if generate_tikz
%     figFileName="../Plots/"+ sprintf("%04d",MP)+"/MPRR_Data_Distribution";
%     savefig(figFileName);
%     saveas(gcf,figFileName,"jpg");
%     cleanfigure('targetResolution', 20)
%     matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false);
%     %export_fig(figFileName,'-eps');
% end
% 
% % SOOT
% figure
% set(gcf,'color','w');
% histogram(Soot_cycle)
% grid on
% ylabel({'\#Data Points'},'Interpreter','latex')
% xlabel("Soot [mg/m$^3$]",'Interpreter', 'latex')
% title('Soot Training Data Distribution','Interpreter', 'latex')
% 
% if generate_tikz
%     figFileName="../Plots/"+ sprintf("%04d",MP)+"/SOOT_Data_Distribution";
%     savefig(figFileName);
%     saveas(gcf,figFileName,"jpg");
%     cleanfigure('targetResolution', 20)
%     matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false);
%     %export_fig(figFileName,'-eps');
% end


end
%% Normalizing data - ONLY with complete dataset!
[u1_target, u1_min, u1_range] = dataTrainStandardized(DOI_main_cycle');
[u2_target, u2_min, u2_range] = dataTrainStandardized(P2M_cycle');
[u3_target, u3_min, u3_range] = dataTrainStandardized(SOI_main_cycle');
[u4_target, u4_min, u4_range] = dataTrainStandardized(H2_doi_cycle');
[u5_target, u5_min, u5_range] = dataTrainStandardized(IMEP_old'); % feedback of old IMEP

[~, y1_min, y1_range] = dataTrainStandardized(IMEP_cycle');
[y2_target, y2_min, y2_range] = dataTrainStandardized(NOx_cycle');
[~, y3_min, y3_range] = dataTrainStandardized(Soot_cycle');
[~, y4_min, y4_range] = dataTrainStandardized(MPRR_cycle');

utrain_1 = normalize_var(utrain(:,1), ...
    u1_min, u1_range, 'to-scaled');
utrain_2 = normalize_var(utrain(:,2), ...
    u2_min, u2_range, 'to-scaled');
utrain_3 = normalize_var(utrain(:,3), ...
    u3_min, u3_range, 'to-scaled');
utrain_4 = normalize_var(utrain(:,4), ...
    u4_min, u4_range, 'to-scaled');
utrain_5 = normalize_var(utrain(:,5), ...
    y1_min, y1_range, 'to-scaled');
ytrain_1 = normalize_var(ytrain(:,1), ...
    y1_min, y1_range, 'to-scaled');
ytrain_2 = normalize_var(ytrain(:,2), ...
    y2_min, y2_range, 'to-scaled');
ytrain_3 = normalize_var(ytrain(:,3), ...
    y3_min, y3_range, 'to-scaled');
ytrain_4 = normalize_var(ytrain(:,4), ...
    y4_min, y4_range, 'to-scaled');


uval_1 = normalize_var(uval(:,1), ...
    u1_min, u1_range, 'to-scaled');
uval_2 = normalize_var(uval(:,2), ...
    u2_min, u2_range, 'to-scaled');
uval_3 = normalize_var(uval(:,3), ...
    u3_min, u3_range, 'to-scaled');
uval_4 = normalize_var(uval(:,4), ...
    u4_min, u4_range, 'to-scaled');
uval_5 = normalize_var(uval(:,5), ...
    y1_min, y1_range, 'to-scaled');
yval_1 = normalize_var(yval(:,1), ...
    y1_min, y1_range, 'to-scaled');
yval_2 = normalize_var(yval(:,2), ...
    y2_min, y2_range, 'to-scaled');
yval_3 = normalize_var(yval(:,3), ...
    y3_min, y3_range, 'to-scaled');
yval_4 = normalize_var(yval(:,4), ...
    y4_min, y4_range, 'to-scaled');

utest_1 = normalize_var(utest(:,1), ...
    u1_min, u1_range, 'to-scaled');
utest_2 = normalize_var(utest(:,2), ...
    u2_min, u2_range, 'to-scaled');
utest_3 = normalize_var(utest(:,3), ...
    u3_min, u3_range, 'to-scaled');
utest_4 = normalize_var(utest(:,4), ...
    u4_min, u4_range, 'to-scaled');
utest_5 = normalize_var(utest(:,5), ...
    y1_min, y1_range, 'to-scaled');
ytest_1 = normalize_var(ytest(:,1), ...
    y1_min, y1_range, 'to-scaled');
ytest_2 = normalize_var(ytest(:,2), ...
    y2_min, y2_range, 'to-scaled');
ytest_3 = normalize_var(ytest(:,3), ...
    y3_min, y3_range, 'to-scaled');
ytest_4 = normalize_var(ytest(:,4), ...
    y4_min, y4_range, 'to-scaled');

utotal_1 = normalize_var(utotal(:,1), ...
    u1_min, u1_range, 'to-scaled');
utotal_2 = normalize_var(utotal(:,2), ...
    u2_min, u2_range, 'to-scaled');
utotal_3 = normalize_var(utotal(:,3), ...
    u3_min, u3_range, 'to-scaled');
utotal_4 = normalize_var(utotal(:,4), ...
    u4_min, u4_range, 'to-scaled');
utotal_5 = normalize_var(utotal(:,5), ...
    y1_min, y1_range, 'to-scaled');
ytotal_1 = normalize_var(ytotal(:,1), ...
    y1_min, y1_range, 'to-scaled');
ytotal_2 = normalize_var(ytotal(:,2), ...
    y2_min, y2_range, 'to-scaled');
ytotal_3 = normalize_var(ytotal(:,3), ...
    y3_min, y3_range, 'to-scaled');
ytotal_4 = normalize_var(ytotal(:,4), ...
    y4_min, y4_range, 'to-scaled');

hyperparam_dataset = [utotal_1';utotal_2';utotal_3';utotal_4';utotal_5'; ytotal_2']';
%% Dateset Definition
% with IMEP feedback u5
utrain_load = [utrain_1'; utrain_2'; utrain_3'; utrain_4'; utrain_5'];
ytrain_load = [ytrain_1'; ytrain_2'; ytrain_3'; ytrain_4'];

uval_load = [uval_1'; uval_2'; uval_3'; uval_4'; uval_5'];
yval_load = [yval_1'; yval_2'; yval_3'; yval_4'];

utest_load = [utest_1'; utest_2'; utest_3'; utest_4'; utest_5'];
ytest_load = [ytest_1'; ytest_2'; ytest_3'; ytest_4'];

%% Save Datafiles
Data = struct();
Data.label = {'imep'; 'nox'; 'soot'; 'mprr'; ...
    'doi_main'; 'p2m'; 'soi_main'; 'doi_h2'; 'imep_old'};
data = {IMEP_cycle', NOx_cycle', Soot_cycle', MPRR_cycle', ...
    DOI_main_cycle', P2M_cycle', SOI_main_cycle', H2_doi_cycle', IMEP_old'};
for ii = 1:length(data)
    [Data.signal{ii, 1}, Data.mean{ii, 1}, Data.std{ii, 1}] = ...
        standardize_data(data{ii});
end

label = Data.label; mean = Data.mean; std = Data.std; signal = Data.signal;
save(fullfile(['../Results/',savename_data]), 'label', 'mean', 'std', 'signal');

DataSets = struct();
DataSets.utrain_load = utrain_load;
DataSets.ytrain_load = ytrain_load;
DataSets.uval_load = uval_load;
DataSets.yval_load = yval_load;
DataSets.utest_load = utest_load;
DataSets.ytest_load = ytest_load;
DataSets.ratio_train = ratio_train;
DataSets.ratio_val = ratio_val;
save(fullfile(['../Results/',savename_datasets]), 'DataSets');

DataSetsPhys = struct();
DataSetsPhys.utrain = utrain;
DataSetsPhys.ytrain = ytrain;
DataSetsPhys.uval = uval;
DataSetsPhys.yval = yval;
DataSetsPhys.utest = utest;
DataSetsPhys.ytest = ytest;
DataSetsPhys.ratio_train = ratio_train;
DataSetsPhys.ratio_val = ratio_val;
DataSetsPhys.label = Data.label; DataSetsPhys.mean = Data.mean; DataSetsPhys.std = Data.std;
save(fullfile(['../Results/',savename_datasets_phys]), 'DataSetsPhys');


%% Training 
for numHiddenUnits1 = [8]%[8,8,8, 10, 10, 10, 12, 12, 12] % [8,8,16,16] % [4,4,6,6,8,8,16,16] % for loop for grid searhc / to try out different units number within the FF layer IMPORTANT PARAMETER
for LSTMStateNum= [8] % [8,16] % [4,4,6,6,8,8] % for loop for grid searhc / to try out different units number within the recurrent layer IMPORTANT PARAMETER
tic

run_nmbr = run_nmbr + 1;
disp ( ['Measurement Point / Save File Number ', num2str(trainingrun)] );
disp ( ['Grid Search Number Iteration ', num2str(run_nmbr)] );

% Model inputs:
%     -DOIMain
%     -SOI pre
%     -SOI Main
%     -H2 DOI
%     -IMEP (last)
% Model Outputs:
%     -IMEP
%     -NOx
%     -Soot
%     -MPRR

% mat = [u1,u2,u3,u4,u5];
% plotmatrix(mat)

%% Create Newtwork arch + setting / options
numResponses = 4; % y1 y2 y3 y4
featureDimension = 5; % u1 u2 u3 u4 u5 % with feedback imep
% featureDimension = 4; % u1 u2 u3 u4 u5
maxEpochs = 1000; % IMPORTANT PARAMETER
miniBatchSize = 512; % IMPORTANT PARAMETER

% addpath('C:\Users\vasu3\Documents\MATLAB\Examples\R2024a\nnet\TrainBayesianNeuralNetworkUsingBayesByBackpropExample')
% architecture
Networklayer_h2df = [...
    sequenceInputLayer(featureDimension)
    fullyConnectedLayer(16*numHiddenUnits1)
    reluLayer
    fullyConnectedLayer(16*numHiddenUnits1)
    reluLayer
    fullyConnectedLayer(8*numHiddenUnits1)
    reluLayer
    gruLayer(LSTMStateNum,'OutputMode','sequence',InputWeightsInitializer='he',RecurrentWeightsInitializer='he')
    fullyConnectedLayer(8*numHiddenUnits1)
    reluLayer
    fullyConnectedLayer(16*numHiddenUnits1)
    reluLayer
    fullyConnectedLayer(numResponses)
    regressionLayer
    ];

% training options
options_tr = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'SequenceLength','shortest',...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'Verbose',1, ...
    'VerboseFrequency',64,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',250,...
    'LearnRateDropFactor',0.75,...
    'L2Regularization',0.01,...
    'ValidationFrequency',40,...
    'ValidationPatience',3,...
    'InitialLearnRate', 0.0005,...
    'Verbose', true, ...
    'ExecutionEnvironment', 'gpu', ...
    'ValidationData',[{uval_load} {yval_load}],...
    'OutputNetwork','best-validation');


%% training and Saving model data
savename = [sprintf('%04d',MP),'_',sprintf('%04d',numHiddenUnits1),'_',sprintf('%04d',LSTMStateNum),'_',sprintf('%04d',trainingrun),'.mat'];

if do_training == true
    tic
    [h2df_model, h2df_model_infor] = trainNetwork(utrain_load,ytrain_load,Networklayer_h2df,options_tr);
    toc
    ElapsedTime = toc;
    h2df_model_analysis = analyzeNetwork(h2df_model); % analysis including total number of learnable parameters
    h2df_model_infor.ElapsedTime = ElapsedTime;

    save(['../Results/h2df_model_',savename],"h2df_model")
    save(['../Results/h2df_model_info_',savename],"h2df_model_infor")
    save(['../Results/h2df_model_analysis_',savename],"h2df_model_analysis")
else
    load(['../Results/h2df_model_',savename])
    load(['../Results/h2df_model_info_',savename])
    load(['../Results/h2df_model_analysis_',savename])
end

%% performance meta data for grid search etc
analysis.FinalRMSE(run_nmbr,1) = h2df_model_infor.FinalValidationRMSE;
analysis.FinalValidationLoss(run_nmbr,1)  = h2df_model_infor.FinalValidationLoss;
analysis.TotalLearnables(run_nmbr,1) = h2df_model_analysis.TotalLearnables;
analysis.ElapsedTime(run_nmbr,1) = h2df_model_infor.ElapsedTime;
analysis.savename(run_nmbr,1:length(savename)) = savename;
savename

%% Plot Training Results
% mkdir
mkdir('../Plots/'+ sprintf("%04d",MP)',sprintf('%04d',trainingrun))


%if do_training
Opts.fontsize = 21;
Opts.axissize = 21; % Define axis label size
f = figure;
set(gcf, 'color', 'w');
set(gcf, 'Position', [100, 100, 1600, 800]);

% Plot training loss
plot(h2df_model_infor.TrainingLoss, 'Marker', 'o', 'MarkerIndices', 1:10:length(h2df_model_infor.TrainingLoss), 'LineWidth', 1.5)
hold on

% Plot validation loss
val_loss = fillmissing(h2df_model_infor.ValidationLoss,'linear');
plot(val_loss, 'Marker', '*', 'MarkerIndices', 1:10:length(h2df_model_infor.TrainingLoss),'LineWidth', 1.5)

% Add best validation point
best_iteration = h2df_model_infor.OutputNetworkIteration;
best_val_loss = val_loss(best_iteration);
plot(best_iteration, best_val_loss, 'pentagram', 'MarkerSize', 15, ...
    'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'red', 'LineWidth', 2)

grid on
set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
xlabel("\#Iterations / -", 'Interpreter', 'latex', 'FontSize', Opts.fontsize)
ylabel("Loss / -", 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
legend(["Training Loss", "Validation Loss", "Best Validation Network"], ...
    'Interpreter', 'latex', 'FontSize', 21);
xTicks = 0:100:400;
set(gca, 'XTick', xTicks);
if generate_tikz
    figFileName="../Plots/"+ sprintf("%04d",MP)+'/'+sprintf('%04d',trainingrun)+"/Loss";
    savefig(figFileName);
    saveas(gcf, figFileName, "jpg");
    cleanfigure('targetResolution', 20)
    matlab2tikz(convertStringsToChars(figFileName+'.tex'), 'showInfo', false);
    exportgraphics(f, strcat(figFileName, '.pdf'), 'BackgroundColor', 'none', 'ContentType', 'vector')
end
%end
%% Prediction on val dataset
y_hat = predict(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4';uval_5']) ; % with IMEP
% feedback
% y_hat = predict(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4']) ;
y1_hat = y_hat(1,:);
y2_hat = y_hat(2,:);
y3_hat = y_hat(3,:);
y4_hat = y_hat(4,:);

% Destandardized Predictions
IMEP_cycle_hat = dataTraindeStandardized(y1_hat,y1_min,y1_range);
NOx_cycle_hat = dataTraindeStandardized(y2_hat,y2_min,y2_range);
Soot_cycle_hat = dataTraindeStandardized(y3_hat,y3_min,y3_range);
MPRR_cycle_hat = dataTraindeStandardized(y4_hat,y4_min,y4_range);

%% Explainability on val dataset
if plot_explainability
    Opts.fontsize = 44;     % Overall axis label font size
    Opts.axissize = 24;     % Axis tick label font size
    Opts.xticksize = 20;    % Specific size for x-tick labels
    
    figure
    set(gcf,'color','w');
    set(gcf, 'Position', [100, 100, 1800, 800]);
    set(gca, 'FontSize', Opts.axissize)

    act = activations(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4';uval_5'],"gru");

    % Create heatmap with additional customization
    h = heatmap(act{1,1}(1:8,1:12), ...
        'CellLabelColor', 'none', 'FontSize', Opts.fontsize);

    % Manually set x and y labels with LaTeX interpreter
    h.XLabel = {"#Cycles / -"};
    h.YLabel = {"#Hidden Unit / -"};
    
    % Customize font sizes
    % h.FontSize = Opts.fontsize;

    % Filename and saving operations remain the same
    figFileName="../Plots/"+ sprintf("%04d",MP)+"/GRUactivaionsval";
    savefig(figFileName);
    saveas(gcf,figFileName,"jpg");
    matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false);
    export_fig(figFileName,'-eps');
    exportgraphics(h,strcat(figFileName, '.pdf'),'BackgroundColor','none','ContentType','vector')
end
%% plotting data on val dataset
if plot_pred_val
    Opts.fontsize = 20; % Font size for labels and titles
    Opts.axissize = 16; % Font size for axis tick labels

    % Determine plot range based on opts_val_small
    if plot_val_small
        plot_range = 1:5000;
        figHandle = figure('Position', [200, 200, 1600, 1000], 'Color', 'w');
    else
        plot_range = 1:size(yval, 1);
        figHandle = figure('Position', [200, 200, 1600, 1000], 'Color', 'w');
    end

    % Set up panel
    p = panel();
    p.pack(4,1); % 4 rows, 1 column
    p.margin = [35 22 12 12]; % Set margins
    p.de.margin = 8; % Space between plots

    % First subplot: IMEP
    p(1,1).select();
    hold on;
    set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
    plot(yval(plot_range,1), 'DisplayName', 'Measured', 'Color', [0.85, 0.325, 0.098]);
    plot(IMEP_cycle_hat(plot_range), 'k-', 'DisplayName', 'Predicted', 'Color', 'k');
    ylabel({'IMEP';'/ Pa'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
    legend('show', 'Location', 'southeast', 'Orientation', 'horizontal', 'FontSize', Opts.axissize, 'Interpreter', 'latex');
    set(gca,'XTickLabel',[])
    grid on;
    box on;
    ax = gca;
    xTicks = 0:1000:5000;
    set(gca, 'XTick', xTicks);
    ax.XRuler.Exponent = 0;
    rmseIMEP_val = rmse(yval(:,1)',IMEP_cycle_hat(1:end),"all");
    target_range_IMEP_val = max(yval(:,1)) - min(yval(:,1));
    rmspeIMEP_val = ((rmseIMEP_val / target_range_IMEP_val)) * 100;

    % Second subplot: NOx
    p(2,1).select();
    hold on;
    set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
    plot(yval(plot_range,2), 'DisplayName', 'Measured', 'Color', [0.85, 0.325, 0.098]);
    plot(NOx_cycle_hat(plot_range), 'k-', 'DisplayName', 'Predicted', 'Color', 'k');
    ylabel({'NOx';'/ ppm'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
    set(gca,'XTickLabel',[])
    grid on;
    box on;
    ax = gca;
    xTicks = 0:1000:5000;
    set(gca, 'XTick', xTicks);
    ax.XRuler.Exponent = 0;
    rmseNOx_val = rmse(yval(:,2)', NOx_cycle_hat(1:end),"all");
    target_range_NOx_val = max(yval(:,2)) - min(yval(:,2));
    rmspeNOx_val = ((rmseNOx_val / target_range_NOx_val)) * 100;

    % Third subplot: Soot
    p(3,1).select();
    hold on;
    set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
    plot(yval(plot_range,3), 'DisplayName', 'Measured', 'Color', [0.85, 0.325, 0.098]);
    plot(Soot_cycle_hat(plot_range), 'k-', 'DisplayName', 'Predicted', 'Color', 'k');
    ylabel({'Soot';'/ mg/m$^3$'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
    set(gca,'XTickLabel',[])
    grid on;
    box on;
    ax = gca;
    ax.XRuler.Exponent = 0;
    xTicks = 0:1000:5000;
    set(gca, 'XTick', xTicks);
    rmseSOOT_val = rmse(yval(:,3)',Soot_cycle_hat(1:end),"all");
    target_range_SOOT_val = max(yval(:,3)) - min(yval(:,3));
    rmspeSOOT_val = ((rmseSOOT_val / target_range_SOOT_val)) * 100;

    % Fourth subplot: MPRR
    p(4,1).select();
    hold on;
    set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
    plot(yval(plot_range,4), 'DisplayName', 'Measured', 'Color', [0.85, 0.325, 0.098]);
    plot(MPRR_cycle_hat(plot_range), 'k-', 'DisplayName', 'Predicted', 'Color', 'k');
    xlabel('$\#$Cycles / -', 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
    ylabel({'MPRR';'/ Pa/0.1 CA$^\circ$'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
    grid on;
    box on;
    ax = gca;
    ax.XRuler.Exponent = 0;
    xTicks = 0:1000:5000;
    set(gca, 'XTick', xTicks);
    rmseMPRR_val = rmse(yval(:,4)',MPRR_cycle_hat(1:end),"all");
    target_range_MPRR_val = max(yval(:,4)) - min(yval(:,4));
    rmspeMPRR_val = ((rmseMPRR_val / target_range_MPRR_val)) * 100;

    % Save the figure in various formats
    figFileName = "../Plots/" + sprintf("%04d", MP) + '/' + sprintf('%04d', trainingrun) + "/Training_Results_Val";
    if generate_tikz
        savefig(figFileName);
        exportgraphics(figHandle, figFileName + ".jpg", 'Resolution', 300, 'BackgroundColor', [1, 1, 1]);
        exportgraphics(figHandle, figFileName + ".pdf", 'BackgroundColor', 'none', 'ContentType', 'vector');
        cleanfigure('targetResolution', 300);
    end

    %rmspe_metrics_val = [rmspeIMEP_val, rmspeNOx_val, rmspeSOOT_val, rmspeMPRR_val];
end

if plot_pred_test
%% Prediction on test dataset
y_hat_tst = predict(h2df_model,[utest_1'; utest_2'; utest_3'; utest_4'; utest_5']) ; % with IMEP
% feedback
% y_hat = predict(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4']) ;
y1_hat_tst = y_hat_tst(1,:);
y2_hat_tst = y_hat_tst(2,:);
y3_hat_tst = y_hat_tst(3,:);
y4_hat_tst = y_hat_tst(4,:);

% Destandardized Predictions
IMEP_cycle_hat_tst = dataTraindeStandardized(y1_hat_tst,y1_min,y1_range);
NOx_cycle_hat_tst = dataTraindeStandardized(y2_hat_tst,y2_min,y2_range);
Soot_cycle_hat_tst = dataTraindeStandardized(y3_hat_tst,y3_min,y3_range);
MPRR_cycle_hat_tst = dataTraindeStandardized(y4_hat_tst,y4_min,y4_range);

hold off
%% Explainability on test dataset
if plot_explainability
    Opts.fontsize = 48; % Overall axis label font size
    Opts.axissize = 48; % Axis tick label font size
    Opts.xticksize = 20; % Specific size for x-tick labels
    Opts.legendsize = 48; % Specific legend font size

    % First figure - Activation Values (in blues)
    figActivations = figure('Position', [100, 100, 1800, 900], 'Color', 'w');
    hold on;
    
    % Get activations
    act = activations(h2df_model,[utest_1'; utest_2'; utest_3'; utest_4';utest_5'],"gru");
    activation_data = act{1,1}(1:8,3:14);
    
    % Generate shades of blue
    blues = [
        0.0000, 0.4470, 0.7410;  % Default MATLAB blue
        0.3010, 0.7450, 0.9330;  % Light blue
        0.0000, 0.3176, 0.6196;  % Medium blue
        0.0000, 0.2470, 0.5410;  % Darker blue
        0.0000, 0.1840, 0.4196;  % Even darker blue
        0.0392, 0.1490, 0.3450;  % Deep blue
        0.0784, 0.1137, 0.2706;  % Very deep blue
        0.1176, 0.0784, 0.1961   % Darkest blue
    ];
    
    % Plot each activation unit
    x_values = 1:12;
    for i = 1:8
        plot(x_values, activation_data(i,:), 'LineWidth', 1.5, ...
            'Color', blues(i,:));
    end
    
    % Customize plot appearance
    grid on;
    box on;
    xlim([1 12]);
    
    % Set axis labels with LaTeX interpreter
    xlabel('$\#$Cycles / -', 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
    ylabel('GRU States / -', 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
    
    % Configure axis properties
    ax = gca;
    set(ax, 'FontSize', Opts.axissize, ...
        'TickLabelInterpreter', 'latex', ...
        'FontName', 'Times New Roman');
    ax.XRuler.Exponent = 0;

    % Second figure - Input signals (keep original colors)
    figSignals = figure('Position', [100, 100, 1800, 900], 'Color', 'w');
    hold on;
    
    % Plot with consistent styling
    p1 = plot(x_values, utest_1(3:14), 'LineWidth', 1.5, 'Color', [0.85, 0.325, 0.098], ...
        'DisplayName', 'DOI Main', 'Marker', 'o', 'MarkerSize', 8);
    p2 = plot(x_values, utest_2(3:14), 'LineWidth', 1.5, 'Color', 'k', ...
        'DisplayName', 'P2M', 'Marker', '*', 'MarkerSize', 8);
    p3 = plot(x_values, utest_3(3:14), 'LineWidth', 1.5, 'Color', [0.929, 0.694, 0.125], ...
        'DisplayName', 'SOI Main', 'Marker', 's', 'MarkerSize', 8);
    p4 = plot(x_values, utest_4(3:14), 'LineWidth', 1.5, 'Color', [0.494, 0.184, 0.556], ...
        'DisplayName', 'DOI H2', 'Marker', 'd', 'MarkerSize', 8);
    p5 = plot(x_values, utest_5(3:14), 'LineWidth', 1.5, 'Color', [0.466, 0.674, 0.188], ...
        'DisplayName', 'Last IMEP', 'Marker', '^', 'MarkerSize', 8);

    % Set axis limits
    xlim([1 12]);
    
    % Customize plot appearance
    grid on;
    box on;
    
    % Set axis labels with LaTeX interpreter
    xlabel('$\#$Cycles / -', 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
    ylabel('Normalized Signal / -', 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
    
    % Configure axis properties
    ax = gca;
    set(ax, 'FontSize', Opts.axissize + 3, ...
        'TickLabelInterpreter', 'latex', ...
        'FontName', 'Times New Roman');
    ax.XRuler.Exponent = 0;
    
    % Add legend with consistent styling
    lgd = legend([p1, p2, p3, p4, p5], ...
        'Location', 'northeast', ...
        'Orientation', 'vertical', ...
        'FontSize', Opts.legendsize, ...
        'Interpreter', 'latex', ...
        'NumColumns', 1);
    lgd.Position = [0.80 0.60 0.15 0.35];

    % Save figures
    figFileName = "../Plots/" + sprintf("%04d",MP) + "/GRUactivaionstest";
    
    % Save activation plot
    figure(figActivations)
    savefig(figFileName + "_activations");
    exportgraphics(gcf, figFileName + "_activations.pdf", 'BackgroundColor', 'none', 'ContentType', 'vector');
    
    % Save input signals plot
    figure(figSignals)
    savefig(figFileName + "_signals");
    exportgraphics(gcf, figFileName + "_signals.pdf", 'BackgroundColor', 'none', 'ContentType', 'vector');
    if generate_tikz
        cleanfigure('targetResolution', 300);
        matlab2tikz(convertStringsToChars(figFileName + "_signals.tex"), 'showInfo', false);
    end
end
%% Plotting on test dataset
Opts.fontsize = 24; % Font size for labels and titles
Opts.axissize = 18; % Font size for axis tick labels
% Create figure with specified dimensions
%figHandle = figure('Position', [200, 200, 800, 1400], 'Color', 'w');
% Set up panel
% Determine the range based on plot_test_small
if plot_test_small
 plot_range = 1:1500;
 figHandle = figure('Position', [200, 200, 800, 1400], 'Color', 'w');
else
 plot_range = 1:size(ytest, 1);
 figHandle = figure('Position', [200,200,1600, 1000], 'Color', 'w');
end
p = panel();
p.pack(4,1); % 4 rows, 1 column
p.margin = [35 22 12 12]; % Set margins
p.de.margin = 8; % Space between plots
% First subplot: IMEP
p(1,1).select();
hold on;
set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
plot(ytest(plot_range,1), 'DisplayName', 'Measured','Color', [0.85, 0.325, 0.098]);
plot(IMEP_cycle_hat_tst(plot_range), 'k-', 'DisplayName', 'Predicted','Color', 'k');
ylabel({'IMEP';'/ Pa'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
legend('show', 'Location', 'southeast', 'Orientation', 'horizontal', 'FontSize', Opts.axissize, 'Interpreter', 'latex');
set(gca,'XTickLabel',[])
grid on;
box on;
ax = gca;
ax.XRuler.Exponent = 0;
xTicks = 0:1000:5000;
set(gca, 'XTick', xTicks);
rmseIMEP_test = rmse(ytest(:,1)',IMEP_cycle_hat_tst(1:end),"all")
target_range_IMEP_test = max(ytest(:,1)) - min(ytest(:,1));
rmspeIMEP_test = ((rmseIMEP_test / target_range_IMEP_test)) * 100
% Second subplot: NOx
p(2,1).select();
hold on;
set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
plot(ytest(plot_range,2),  'DisplayName', 'Measured','Color', [0.85, 0.325, 0.098]);
plot(NOx_cycle_hat_tst(plot_range), 'k-', 'DisplayName', 'Predicted','Color', 'k');
ylabel({'NOx';'/ ppm'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
%legend('show', 'Location', 'southeast', 'Orientation', 'horizontal', 'FontSize', Opts.axissize, 'Interpreter', 'latex');
set(gca,'XTickLabel',[])
xTicks = 0:1000:5000;
set(gca, 'XTick', xTicks);
grid on;
box on;
ax = gca;
ax.XRuler.Exponent = 0;
rmseNOx_test = rmse(ytest(:,2)',NOx_cycle_hat_tst(1:end),"all")
target_range_NOx_test = max(ytest(:,2)) - min(ytest(:,2));
rmspeNOx_test = ((rmseNOx_test / target_range_NOx_test)) * 100
% Third subplot: Soot
p(3,1).select();
hold on;
set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
plot(ytest(plot_range,3), 'DisplayName', 'Measured','Color', [0.85, 0.325, 0.098]);
plot(Soot_cycle_hat_tst(plot_range), 'k-', 'DisplayName', 'Predicted','Color', 'k');
ylabel({'Soot';'/ mg/m$^3$'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
%legend('show', 'Location', 'southeast', 'Orientation', 'horizontal', 'FontSize', Opts.axissize, 'Interpreter', 'latex');
set(gca,'XTickLabel',[])
xTicks = 0:1000:5000;
set(gca, 'XTick', xTicks);
grid on;
box on;
ax = gca;
ax.XRuler.Exponent = 0;
rmseSoot_test = rmse(ytest(:,3)',Soot_cycle_hat_tst(1:end),"all")
target_range_Soot_test = max(ytest(:,3)) - min(ytest(:,3));
rmspeSoot_test = ((rmseSoot_test / target_range_Soot_test)) * 100
% Fourth subplot: MPRR
p(4,1).select();
hold on;
set(gca, 'FontSize', Opts.axissize, 'TickLabelInterpreter', 'latex');
plot(ytest(plot_range,4), 'DisplayName', 'Measured','Color', [0.85, 0.325, 0.098]);
plot(MPRR_cycle_hat_tst(plot_range), 'k-', 'DisplayName', 'Predicted','Color', 'k');
xlabel('$\#$Cycles / -', 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
ylabel({'MPRR';'/ Pa/0.1 CA$^\circ$'}, 'Interpreter', 'latex', 'FontSize', Opts.fontsize);
%legend('show', 'Location', 'southeast', 'Orientation', 'horizontal', 'FontSize', Opts.axissize, 'Interpreter', 'latex');
grid on;
box on;
ax = gca;
ax.XRuler.Exponent = 0;
xTicks = 0:1000:5000;
set(gca, 'XTick', xTicks);
rmseMPRR_test = rmse(ytest(:,4)',MPRR_cycle_hat_tst(1:end),"all")
target_range_MPRR_test = max(ytest(:,4)) - min(ytest(:,4));
rmspeMPRR_test = ((rmseMPRR_test / target_range_MPRR_test)) * 100
% Save the figure in various formats
figFileName = "../Plots/" + sprintf("%04d", MP) + '/' + sprintf('%04d', trainingrun) + "/Training_Results_Test";
if generate_tikz
 savefig(figFileName);
 exportgraphics(figHandle, figFileName + ".jpg", 'Resolution', 300, 'BackgroundColor', [1, 1, 1]);
 exportgraphics(figHandle, figFileName + ".pdf", 'BackgroundColor', 'none', 'ContentType', 'vector');
 cleanfigure('targetResolution', 300);
%matlab2tikz(strcat(figFileName, '.tex'), 'showInfo', false);
end
rmspe_metrics_val = [rmspeIMEP_val, rmspeNOx_val, rmspeSOOT_val, rmspeMPRR_val]
rmspe_metrics_test = [rmspeIMEP_test, rmspeNOx_test ,rmspeSoot_test,rmspeMPRR_test]

end


%% Define prediction function- we can use it later for MPC

% We have 8 hidden states 
unit_size = LSTMStateNum;
% Use aliases for network parameters

% Fully connected layers- 1
WFc1 =  h2df_model.Layers(2, 1).Weights;
bFc1 =  h2df_model.Layers(2, 1).Bias;

% Fully connected layers- 2
WFc2 =  h2df_model.Layers(4, 1).Weights;
bFc2 =  h2df_model.Layers(4, 1).Bias;

% Fully connected layers- 3
WFc3 =  h2df_model.Layers(6, 1).Weights;
bFc3 =  h2df_model.Layers(6, 1).Bias;


% Recurrent weights
Rr =  h2df_model.Layers(8, 1).RecurrentWeights(1:unit_size, :);
Rz=  h2df_model.Layers(8, 1).RecurrentWeights(unit_size+1:2*unit_size, :);
Rh =  h2df_model.Layers(8, 1).RecurrentWeights(2*unit_size+1:3*unit_size, :);
%Ro =  h2df_model.Layers(8, 1).RaecurrentWeights(3*unit_size+1:end, :);

% Input weights
wr =  h2df_model.Layers(8, 1).InputWeights(1:unit_size, :);
wz =  h2df_model.Layers(8, 1).InputWeights(unit_size+1:2*unit_size, :);
wh =  h2df_model.Layers(8, 1).InputWeights(2*unit_size+1:3*unit_size, :);
%wo =  h2df_model.Layers(8, 1).InputWeights(3*unit_size+1:end, :);

% Bias weights
br =  h2df_model.Layers(8, 1).Bias(1:unit_size, :);
bz =  h2df_model.Layers(8, 1).Bias(unit_size+1:2*unit_size, :);
bh =  h2df_model.Layers(8, 1).Bias(2*unit_size+1:3*unit_size, :);
%bo =  h2df_model.Layers(8, 1).Bias(3*unit_size+1:end, :);

% Fully connected layers- 4
WFc4 =  h2df_model.Layers(9, 1).Weights;
bFc4 =  h2df_model.Layers(9, 1).Bias;

% Fully connected layers- 5
WFc5 =  h2df_model.Layers(11, 1).Weights;
bFc5 =  h2df_model.Layers(11, 1).Bias;

% Fully connected layers- 6
WFc6 =  h2df_model.Layers(13, 1).Weights;
bFc6 =  h2df_model.Layers(13, 1).Bias;

%% Assigining parameters to a structure

Par.Rz = double(Rz);
Par.Rr = double(Rr);
Par.Rh = double(Rh);
%Par.Ro = double(Ro);
Par.wz = double(wz);
Par.wr = double(wr);
Par.wh = double(wh);
%Par.wo = double(wo);
Par.bz = double(bz);
Par.br = double(br);
Par.bh = double(bh);
%Par.bo = double(bo);
Par.WFc1 = double(WFc1);
Par.bFc1 = double(bFc1);
Par.WFc2 = double(WFc2);
Par.bFc2 = double(bFc2);
Par.WFc3 = double(WFc3);
Par.bFc3 = double(bFc3);
Par.WFc4 = double(WFc4);
Par.bFc4 = double(bFc4);
Par.WFc5 = double(WFc5);
Par.bFc5 = double(bFc5);
Par.WFc6 = double(WFc6);
Par.bFc6 = double(bFc6);
Par.nCellStates = 0; %change for lstm to hiddenstates (Par.nHiddenStates
Par.nHiddenStates = unit_size;
Par.nStates = Par.nHiddenStates;
Par.nActions = featureDimension;
Par.nOutputs = numResponses; 
Par.TotalLearnables = analysis.TotalLearnables(run_nmbr,1) ;
Par.FinalRMSE = analysis.FinalRMSE(run_nmbr,1);
Par.FinalValidationLoss = analysis.FinalValidationLoss(run_nmbr,1);
Par.ElapsedTime = analysis.ElapsedTime(run_nmbr,1);
Par.Savename = analysis.savename(run_nmbr,1);
% Par.RMSE_Test = [rmseIMEP_tst, rmseNOx_tst, rmseSOOT_tst, rmseMPRR_tst];
% Par.RMSE_Val = [rmseIMEP_val, rmseNOx_val, rmseSOOT_val, rmseMPRR_val];
% Par.RMSPE_Test = [rmspeIMEP_tst, rmspeNOx_tst, rmspeSOOT_tst, rmspeMPRR_tst];
% Par.RMSPE_Val = [rmspeIMEP_val, rmspeNOx_val, rmspeSOOT_val, rmspeMPRR_val];

if do_training == true
    save(['../Results/Par_',savename],"Par")
end

trainingrun = trainingrun + 1; % increase when doing grid search

if break_loop break
end

end

if break_loop break
end

end

%% verfiy own function
if verify_my_func

% %% Simulating your model
% 
% 
uts = [uval_1';uval_2';uval_3';uval_4';uval_5'];
xt1 = zeros(Par.nStates,1);
y1_hat_myfnc = zeros(length(uval_1),1);
y2_hat_myfnc = zeros(length(uval_1),1);
y3_hat_myfnc = zeros(length(uval_1),1);
y4_hat_myfnc = zeros(length(uval_1),1);

for i = 1:length(uval_1)

%[xt,y] = MyLSTMstateFnc(xt1, uts(:,i),Par);
[xt,y] = MyGRUstateFnc(xt1, uts(:,i),Par);

y1_hat_myfnc(i,1) = y(1);
y2_hat_myfnc(i,1) = y(2);
y3_hat_myfnc(i,1) = y(3);
y4_hat_myfnc(i,1) = y(4);
xt1 = xt;


end



figure(15) % comparing mylstmstatefun with matlab prediction - this should be the same!
set(gcf, 'Position', [100, 100, 900, 1200]);
set(gcf,'color','w');


subplot(4,1,1)
plot(y1_hat)
hold on
plot(y1_hat_myfnc, 'r--')
grid on
ylabel('IMEP','Interpreter','latex')
set(gca,'FontSize',14)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
legend({'Predicted IMEP model','Predicted IMEP function'},'Location','southeast','Orientation','horizontal')


subplot(4,1,2)
plot(y2_hat)
hold on
plot(y2_hat_myfnc, 'r--')
grid on
ylabel('NOx','Interpreter','latex')
set(gca,'FontSize',14)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
legend({'Predicted NOx model','Predicted NOx function'},'Location','southeast','Orientation','horizontal')

subplot(4,1,3)
plot(y3_hat)
hold on
plot(y3_hat_myfnc, 'r--')
grid on
ylabel('Soot','Interpreter','latex')
set(gca,'FontSize',14)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
legend({'Predicted Soot model','Predicted Soot function'},'Location','southeast','Orientation','horizontal')

subplot(4,1,4)
plot(y4_hat)
hold on
plot(y4_hat_myfnc, 'r--')
grid on
ylabel('MPRR','Interpreter','latex')
set(gca,'FontSize',14)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
legend({'Predicted MPRR model','Predicted MPRR function'},'Location','southeast','Orientation','horizontal')
end