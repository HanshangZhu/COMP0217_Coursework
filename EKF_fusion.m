clc; clear;

%% === 加载数据和标定参数 ===
load('calibration_params.mat');             % 包含 gyro_bias, accel_bias, tof_bias
load('trainingData/task2_3.mat');           % 包含 out 结构体

%% === 传感器数据提取 ===
gyro  = squeeze(out.Sensor_GYRO.signals.values)';   % [N x 3]
accel = squeeze(out.Sensor_ACCEL.signals.values)';  % [N x 3]

tof1_data = squeeze(out.Sensor_ToF1.signals.values);
tof2_data = squeeze(out.Sensor_ToF2.signals.values);
tof3_data = squeeze(out.Sensor_ToF3.signals.values);
tof_data = [tof3_data(:,1), tof2_data(:,1), tof1_data(:,1)];

time    = out.GT_time.time;
GT_pos  = out.GT_position.signals.values(:,1:2);
GT_rot  = out.GT_rotation.signals.values;
GT_rot(1,:) = GT_rot(2,:);  % 修复 NaN

GT_q = quaternion(GT_rot);
eul  = eulerd(GT_q, 'ZYX', 'frame');
GT_yaw = deg2rad(eul(:,1)) + pi;
GT_yaw = unwrap(GT_yaw);
init_cor = GT_pos(1,:);
init_rot = GT_yaw(1);

%% === Bias 标定 ===
gyro_calibrated  = gyro - calibration_params.gyro_bias;
accel_calibrated = accel - calibration_params.accel_bias;
tof_calibrated   = tof_data - calibration_params.tof_bias;

%% === 突然刹车检测 ===
accel_dz = [0; diff(accel_calibrated(:,3))];
thresh = 0;
is_brake_jump = (accel_dz < -thresh) & (accel_calibrated(:,3) < 0);

%% === 滑动窗口静止检测参数 ===
window_size = 8;     % 滑动窗口大小（单位：帧）
acc_thresh = 0.2;    % 加速度模长平均值阈值（单位：m/s²）

%% === EKF 初始化 ===
fs = 1 / mean(diff(time));
N = length(time);
x_est = zeros(N, 5);  % [x, y, vx, vy, yaw]
x_est(1,:) = [init_cor, 0, 0, init_rot];
P = diag([0.01, 0.01, 0.01, 0.01, 0]);
Q = diag([0.0001, 0.0001, 0.001, 0.001, 0]);
R = eye(3) * 1e17;

dt = 1 / fs;
a_w = zeros(N,2);
for i = 2:N
    %% --- 预测步 ---
    a_y = accel_calibrated(i,2);
    a_z = accel_calibrated(i,3);

    if is_brake_jump(i)
        scale = 1.3;
    else
        scale = 1.0;
    end
    a_b = [a_y, a_z * scale];

    yaw = x_est(i-1,5);
    R_yaw = [sin(yaw), cos(yaw); -cos(yaw), sin(yaw)];
    a_w(i,:) = (R_yaw * a_b')';

    vx = x_est(i-1,3) + a_w(i,1) * dt;
    vy = x_est(i-1,4) + a_w(i,2) * dt;

    %% === 滑动窗口判断静止状态，强制速度归零 ===
    if i > window_size
        a_recent = accel_calibrated(i-window_size+1:i, 2:3);  % Y/Z方向
        a_norm_mean = mean(vecnorm(a_recent, 2, 2));  % 模长的平均值
        if a_norm_mean < acc_thresh
            vx = 0;
            vy = 0;
        end
    end

    x_pred = x_est(i-1,:);
    x_pred(1) = x_pred(1) + vx * dt + 0.5 * a_w(i,1) * dt^2;
    x_pred(2) = x_pred(2) + vy * dt + 0.5 * a_w(i,2) * dt^2;
    x_pred(3) = vx;
    x_pred(4) = vy;
    x_pred(5) = x_pred(5) + gyro_calibrated(i,1) * dt;

    F = eye(5); F(1,3) = dt; F(2,4) = dt;
    P = F * P * F' + Q;

    [z_pred, H] = measurementModel(x_pred);
    z_meas = tof_calibrated(i,:)';
    y = z_meas - z_pred;
    S = H * P * H' + R;
    K = P * H' / S;
    x_est(i,:) = x_pred + (K * y)';
    x_est(i,5) = x_pred(5);
    P = (eye(5) - K * H) * P;
end

%% === 可视化 ===
figure;
subplot(3,2,1);
plot(GT_pos(:,1), GT_pos(:,2), 'g--', 'LineWidth', 2); hold on;
plot(x_est(:,1), x_est(:,2), 'r-', 'LineWidth', 1.5);
legend('Ground Truth', 'EKF Estimate');
xlabel('X [m]'); ylabel('Y [m]');
title('Trajectory with EKF Fusion'); axis equal; grid on;

subplot(3,2,2);
plot(time, x_est(:,4), 'r-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Y-axis Speed [m/s]');
title('Y-axis Velocity from EKF'); grid on;

subplot(3,2,3);
plot(time, a_w(:,2), 'b-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Accel Y [m/s^2]');
title('World-frame Y Acceleration (Raw)'); grid on;

subplot(3,2,4);
plot(time, GT_pos(:,1), 'b-', 'LineWidth', 1.2); hold on;
plot(time, GT_pos(:,2), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Position [m]');
legend('GT X', 'GT Y');
title('Ground Truth Position over Time'); grid on;

subplot(3,2,5);
plot(time, x_est(:,1), 'b-', 'LineWidth', 1.2); hold on;
plot(time, x_est(:,2), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Position [m]');
legend('Est X', 'Est Y');
title('Estimated Position over Time'); grid on;

subplot(3,2,6);
plot(time, x_est(:,5), 'r-', 'LineWidth', 1.2); hold on;
plot(time, GT_yaw, 'b--', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Yaw [rad]');
legend('Estimated', 'Ground Truth');
title('Yaw Comparison over Time'); grid on;

sgtitle('EKF Sensor Fusion Visualization');

%% === Measurement Model Function ===
function [z_pred, H] = measurementModel(x)
    x_left = 1.2; x_right = -1.2; y_top = 1.2;
    x_pos = x(1); y_pos = x(2); theta = x(5);
    epsilon = 1e-4;
    sin_theta = sin(theta); cos_theta = cos(theta);
    safe_sin = sign(sin_theta) * max(abs(sin_theta), epsilon);
    safe_sin_sq = safe_sin^2;

    t_left  = ((x_left - x_pos)) / safe_sin;
    t_front = (y_top - y_pos) / safe_sin;
    t_right = -(x_right - x_pos) / safe_sin;

    z_pred = [t_right; t_front; t_left];
    H = zeros(3, 5);
    H(1,1) = 1 / safe_sin;
    H(1,5) = - (x_pos - x_left) * cos_theta / safe_sin_sq;
    H(2,2) = -1 / safe_sin;
    H(2,5) = - (y_top - y_pos) * cos_theta / safe_sin_sq;
    H(3,1) = -1 / safe_sin;
    H(3,5) = - (x_right - x_pos) * cos_theta / safe_sin_sq;
end
