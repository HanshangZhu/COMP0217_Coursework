function [X_Est, P_Est, GT] = myEKF(out, bias, T)
% MYEKF Extended Kalman Filter for 2D position and orientation estimation using IMU and ToF

% ------------------ Extract Sensor Data ------------------ %
accel = squeeze(permute(out.Sensor_ACCEL.signals.values, [3, 2, 1]));
gyro  = squeeze(permute(out.Sensor_GYRO.signals.values, [3, 2, 1]));
mag   = squeeze(permute(out.Sensor_MAG.signals.values, [3, 2, 1]));
tof1  = squeeze(out.Sensor_ToF1.signals.values(:,1,:));
tof2  = squeeze(out.Sensor_ToF2.signals.values(:,1,:));
tof3  = squeeze(out.Sensor_ToF3.signals.values(:,1,:));
pos   = out.GT_position.signals.values;
rot   = out.GT_rotation.signals.values;
timeVec = out.GT_position.time;

N = min([size(accel,1), size(gyro,1), size(mag,1), size(tof1,1), size(timeVec,1)]);
pos   = pos(1:N,:);
rot   = rot(1:N,:);
timeVec = timeVec(1:N);

% ------------------ Frame Convention ------------------ %
% Separate rotation matrices for accelerometer/magnetometer and gyroscope
Rwb_accel = [0 1 0; 
             0 0 1; 
            -1 0 0];

Rwb_mag  = [0 1 0; 
            0 0 -1; 
            1 0 0];



accel_corrected = (Rwb_accel * accel')';
gyro_corrected  = (Rwb_accel * gyro')';
mag_calibrated = (mag - bias') * T;

% Scale to match Earth's field magnitude (~48µT in London)
H_ref = 48e-6;

mag_calibrated = H_ref * (mag_calibrated ./ vecnorm(mag_calibrated, 2, 2));

mag_corrected = (Rwb_mag * mag_calibrated')';  % Rotate to world frame

% ------------------ Initialisation ------------------ %
X_Est = zeros(N, 6); % [x y vx vy psi dpsi]
P_Est = cell(N,1);
GT = pos;

X_Est(1,:) = zeros(1,6);
P_Est{1} = eye(6)*0.01;

Q = diag([1e-6, 1e-6, 1e-3, 1e-3, 1e-6, 1e-3]);
R = diag([(1*pi/180)^2, (1*pi/180)^2,  0.05^2, 0.05^2, 0.05^2]);

for k = 1:N-1
    dt = timeVec(k+1) - timeVec(k);
    if dt <= 0, dt = 1e-2; end

    xk = X_Est(k,:)';
    Pk = P_Est{k};

    af = accel_corrected(k,3);
    as = -accel_corrected(k,1);
    gy = gyro_corrected(k,3);
    psi = xk(5);
    c = cos(psi); s = sin(psi);
    ax_w = c*af - s*as;
    ay_w = s*af + c*as;

    x_pred = [xk(1) + xk(3)*dt + 0.5*ax_w*dt^2;
              xk(2) + xk(4)*dt + 0.5*ay_w*dt^2;
              xk(3) + ax_w*dt;
              xk(4) + ay_w*dt;
              xk(5) + xk(6)*dt;
              gy];

    dax_dpsi = -s*af - c*as;
    day_dpsi =  c*af - s*as;

    F = eye(6);
    F(1,3) = dt;
    F(1,5) = 0.5*dt^2*dax_dpsi;
    F(2,4) = dt;
    F(2,5) = 0.5*dt^2*day_dpsi;
    F(3,5) = dt*dax_dpsi;
    F(4,5) = dt*day_dpsi;
    F(5,6) = dt;

    P_pred = F*Pk*F' + Q;

    psi_meas = atan2(mag_corrected(k,1), mag_corrected(k,3));
    dpsi_meas = gy;
    z_meas = [psi_meas; dpsi_meas; tof1(k); tof2(k); tof3(k)];
    z_pred = [x_pred(5); x_pred(6); 0; 0; 0];

    H = zeros(5,6);
    H(1,5) = 1;
    H(2,6) = 1;

    y = z_meas - z_pred;
    S = H*P_pred*H' + R;
    K = P_pred*H'/S;

    x_upd = x_pred + K*y;
    P_upd = (eye(6) - K*H)*P_pred;

    X_Est(k+1,:) = x_upd';
    P_Est{k+1} = P_upd;
end
end
