function [X_Est, P_Est, GT] = myEKF(out)
    %% Sensor Data Extraction from Simulation Output
    accel = squeeze(permute(out.Sensor_ACCEL.signals.values, [3, 2, 1]));
    gyro  = squeeze(permute(out.Sensor_GYRO.signals.values, [3, 2, 1]));
    mag   = squeeze(permute(out.Sensor_MAG.signals.values, [3, 2, 1]));
    tof_front = squeeze(out.Sensor_ToF1.signals.values(:,1,:));
    tof_left  = squeeze(out.Sensor_ToF2.signals.values(:,1,:));
    tof_right = squeeze(out.Sensor_ToF3.signals.values(:,1,:));
    pos   = out.GT_position.signals.values;
    timeVec = out.GT_position.time;
    sensor_calibration = load("sensor_calibration.mat");

    %% Time Alignment and Truncation
    N = min([size(accel,1), size(gyro,1), size(mag,1), length(tof_front), length(timeVec)]);
    pos = pos(1:N,:);
    timeVec = timeVec(1:N);
    GT = pos;

    %% Frame Convention
    accel = (sensor_calibration.Rotw_a * accel')';
    gyro = (sensor_calibration.Rotw_a * gyro')';
    mag = (sensor_calibration.Rotw_mag * mag')';

    %% Sensor Calibration
    accel = accel - sensor_calibration.accel_bias;
    gyro = gyro - sensor_calibration.gyro_bias;
    mag = (mag - sensor_calibration.b_mag) * sensor_calibration.A_mag;

    %% EKF Initialization
    X_Est = zeros(N, 8);
    P_Est = cell(N,1);
    X_Est(1,:) = [ pos(1,1:2), 0, 0, 0, 0, 0, 0 ];
    P_Est{1} = diag([0.01, 0.01, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001]);
    Q = diag([1e-6, 1e-6, 1e-3, 1e-3, 1e-6, 1e-3, 1e-8, 1e-8]);
    R = diag([sensor_calibration.R_psi, sensor_calibration.R_tof_left, sensor_calibration.R_tof_middle, sensor_calibration.R_tof_right]);

    %% Main EKF Loop
    for k = 1:N-1
        dt = timeVec(k+1) - timeVec(k);
        if dt <= 0, dt = 1e-2; end

        xk = X_Est(k,:)';
        [x_pred, F] = predictionStepSymbolic(xk, accel(k,:), gyro(k,:), dt);
        [z_meas, H] = measurementModelOptim(x_pred, mag(k,:), tof_front(k), tof_left(k), tof_right(k));
        [x_upd, P_upd] = updateStep(x_pred, P_Est{k}, z_meas, H, R);

        X_Est(k+1,:) = x_upd';
        P_Est{k+1} = P_upd;
    end
end

function [x_pred, F] = predictionStepSymbolic(xk, accel, gyro, dt)
    x = xk(1); y = xk(2); vx = xk(3); vy = xk(4);
    psi = xk(5); dpsi = xk(6); gb = xk(7); ab = xk(8);
    af = accel(3) - ab;
    gy = gyro(3) - gb;
    c = cos(psi); s = sin(psi);

    x_pred = zeros(8,1);
    x_pred(1) = x + vx*dt + 0.5*c*af*dt^2;
    x_pred(2) = y + vy*dt + 0.5*s*af*dt^2;
    x_pred(3) = vx + c*af*dt;
    x_pred(4) = vy + s*af*dt;
    x_pred(5) = psi + dpsi*dt;
    x_pred(6) = gy;
    x_pred(7) = gb;
    x_pred(8) = ab;

    F = jacobian_prediction(x, y, vx, vy, psi, dpsi, gb, ab, dt, af);
end

function [z_meas, H] = measurementModelOptim(x_pred, mag, d1, d2, d3)
    psi = x_pred(5);
    psi_meas = atan2(mag(1), mag(3));
    x = x_pred(1); y = x_pred(2);

    offsets = [0, pi/2, -pi/2];
    angles = psi + offsets;
    d_pred = zeros(3,1);
    H = zeros(4,8);

    for i = 1:3
        theta = angles(i);
        [t, ~, dt_dx, dt_dy, dt_dpsi] = rayBoxIntersection(x, y, theta);
        d_pred(i) = t;
        H(i+1,1) = dt_dx;
        H(i+1,2) = dt_dy;
        H(i+1,5) = dt_dpsi;
    end

    z_meas = [psi_meas; d1; d2; d3];
    H(1,5) = 1;
end

function [x_upd, P_upd] = updateStep(x_pred, P_pred, z_meas, H, R)
    psi = x_pred(5);
    x = x_pred(1);
    y = x_pred(2);
    offsets = [0, pi/2, -pi/2];
    angles = psi + offsets;
    d_pred = zeros(3,1);

    for i = 1:3
        d_pred(i) = rayBoxIntersection(x, y, angles(i));
    end

    z_pred = [psi; d_pred];
    y_tilde = z_meas - z_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;
    x_upd = x_pred + K * y_tilde;
    P_upd = (eye(length(x_pred)) - K * H) * P_pred;
end

function [t, branch, dt_dx, dt_dy, dt_dpsi] = rayBoxIntersection(x, y, theta)
    x_left = -1.2; x_right = 1.2;
    y_bottom = -1.2; y_top = 1.2;

    if cos(theta) > 0
        x_wall = x_right;
    elseif cos(theta) < 0
        x_wall = x_left;
    else
        x_wall = NaN;
    end

    if ~isnan(x_wall) && abs(cos(theta)) > 1e-6
        t_v = (x_wall - x) / cos(theta);
    else
        t_v = inf;
    end

    if sin(theta) > 0
        y_wall = y_top;
    elseif sin(theta) < 0
        y_wall = y_bottom;
    else
        y_wall = NaN;
    end

    if ~isnan(y_wall) && abs(sin(theta)) > 1e-6
        t_h = (y_wall - y) / sin(theta);
    else
        t_h = inf;
    end

    if t_v <= 0, t_v = inf; end
    if t_h <= 0, t_h = inf; end

    if t_v < t_h
        t = t_v; branch = 1;
        dt_dx = -1 / cos(theta);
        dt_dy = 0;
        dt_dtheta = (x_wall - x) * sin(theta) / cos(theta)^2;
    else
        t = t_h; branch = 2;
        dt_dx = 0;
        dt_dy = -1 / sin(theta);
        dt_dtheta = - (y_wall - y) * cos(theta) / sin(theta)^2;
    end
    dt_dpsi = dt_dtheta;
end
