
clc
clear

set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
addpath('D:\Dropbox\MATLAB_for_Baslisk')
name_list = {'test_BK_0d23_1d42','test_BK_0d24_1d10','test_BK_0d24_1d46','test_BK_0d224_1d073','test_BK_0d236_1d132','test_BK_0d237_1d091','test_BK_0d239_1d435','test_BK_0d243_1d223','test_BK_0d243_1d223','test_BK_0d243_1d313','test_BK_0d243_1d465','test_BK_0d246_1d068'};
u_in = zeros(1000000,15);
process_num = 1;
tar_eta_full_RHS = zeros(1000000,1);
tar_eta_residual = zeros(1000000,1);
for aas = 1:length(name_list)
    try
        load([name_list{aas} '_bak_surf1_pert.mat']);
    catch
        disp(name_list{aas})
        continue
    end
    fn = fieldnames(sfr3);
    for i = 1:length(fn)
        name = fn{i};
        sfr3.(name) = sfr3.(name)(:,1:time_len_min); % No .Variables needed
    end
% 

t=time_vec;
x=x1;
dx = x(3) - x(2);
eta_smooth=spline_smooth_5(sfr3.eta,t,x);
ux_smooth=spline_smooth_5(sfr3.ux,t,x);
uy_smooth=spline_smooth_5(sfr3.uy,t,x);


deta_dt = -grad1(sfr3.eta,time_vec(2)-time_vec(1));
deta_dt_smooth = spline_smooth_5(deta_dt,t,x);

deta_dx = -grad1(sfr3.eta.',x(2)-x(1)).';
deta_dx_smooth = spline_smooth_5(deta_dx,t,x);


deta_dx2 = -grad1_smooth(deta_dx.',x(2)-x(1)).';
deta_dx2_smooth = spline_smooth_5(deta_dx2,t,x);


ux = squeeze(sfr3.ux);
dudx = -grad1_smooth(ux.',x(2)-x(1)).';
dudx_smooth = spline_smooth_5(dudx,t,x);

uy = squeeze(sfr3.uy);
duydx = -grad1_smooth(uy.',x(2)-x(1)).';
duydx_smooth = spline_smooth_5(duydx,t,x);

duydy_smooth = spline_smooth_5(sfr3.dwd_y,t,x);

ut_x = squeeze(sfr3.dudt_T_x);
du_dt = -grad1_smooth(ux,time_vec(2)-time_vec(1));
du_dt_smooth = spline_smooth_5(du_dt,t,x);


sfr3.eta=spline_smooth_5(sfr3.eta,t,x);
sfr3.ux=spline_smooth_5(sfr3.ux,t,x);
sfr3.uy=spline_smooth_5(sfr3.uy,t,x);
sfr3.dud_x=spline_smooth_5(sfr3.eta,t,x);
sfr3.dud_y=spline_smooth_5(sfr3.dud_y,t,x);
sfr3.dwd_x=spline_smooth_5(sfr3.dwd_x,t,x);
sfr3.dwd_y=spline_smooth_5(sfr3.dwd_y,t,x);
sfr3.dudt_T_x=spline_smooth_5(sfr3.dudt_T_x,t,x);
sfr3.dudt_T_y=spline_smooth_5(sfr3.dudt_T_y,t,x);
sfr3.p=spline_smooth_5(sfr3.p,t,x);

sfr = sfr3;

n_s = split(name_list{aas},'_');
T_s = split(n_s{4},'d');
Tp = str2double(T_s{1})+str2double(['0.' T_s{2}]);
f=1/Tp;
k=(2*pi*f)^2/9.81;
omega = 2*pi*f;

RHS=-9.81*deta_dx_smooth-dudx_smooth.*sfr.ux-(duydx_smooth.*sfr.uy)-sfr.uy.*duydx_smooth.*deta_dx_smooth.^2-sfr.uy.^2.*deta_dx_smooth.*deta_dx2_smooth;


result = du_dt_smooth - RHS;


MSK = abs(result)>=0.04*max(max(abs(result)));
MSK(:,1:200) = false;


eta_x_temp = deta_dx_smooth(MSK);

num_data_points = size(eta_x_temp, 1);
eta_H = hilbert(sfr.eta);
eta_H_x = hilbert(sfr.eta.').';
% Preallocating memory for u_in
residual_full = du_dt_smooth;
residual_res = du_dt_smooth - RHS +-  (-0.40672904 * (dudx_smooth + -0.3146089));
% Assuming other variables (e.g., deta_dx, uy, ux, etc.) are already defined
u_in(process_num: process_num+num_data_points-1,1) = reshape(deta_dx_smooth(MSK), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,2) = reshape(sfr.uy(MSK), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,3) = reshape(sfr.ux(MSK), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,4) = reshape(dudx_smooth(MSK), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,5) = reshape(sfr.dud_y(MSK), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,6) = reshape(duydx_smooth(MSK), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,7) = reshape(sfr.dwd_y(MSK), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,8) = reshape(deta_dx2(MSK), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,9) = reshape(imag(eta_H(MSK)), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,10)= reshape(imag(eta_H_x(MSK)), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,11) = reshape(abs(eta_H(MSK)), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,12) = reshape(abs(eta_H_x(MSK)), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,13) = reshape((eta_smooth(MSK)), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,14) = reshape((eta_smooth(MSK).*0+k), [num_data_points, 1]);
u_in(process_num: process_num+num_data_points-1,15) = reshape((eta_smooth(MSK).*0+omega), [num_data_points, 1]);


tar_eta_full (process_num: process_num+num_data_points-1) = reshape(residual_full(MSK),[num_data_points, 1]);
tar_eta_residual (process_num: process_num+num_data_points-1) = reshape(residual_res(MSK),[num_data_points, 1]);
process_num=process_num+num_data_points;

end
u_in = u_in(1: process_num-1,:);
tar_eta_residual = tar_eta_residual(1: process_num-1);
tar_eta_full = tar_eta_full(1: process_num-1);
formatOut = 'ddmmyy';
t_save = datestr(now,formatOut);
save(['D:\Dropbox\MATLAB_for_Baslisk\db_database_full_discover_' t_save '.mat'])

