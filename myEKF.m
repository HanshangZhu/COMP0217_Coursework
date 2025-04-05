function [X_Est, P_Est, GT] = myEKF(out)
    %% Sensor Data Extraction from Simulation Output
    accel = squeeze(permute(out.Sensor_ACCEL.signals.values, [3, 2, 1]));
    gyro  = squeeze(permute(out.Sensor_GYRO.signals.values, [3, 2, 1]));
    mag   = squeeze(permute(out.Sensor_MAG.signals.values, [3, 2, 1]));
    tof_front = squeeze(out.Sensor_ToF2.signals.values(:,1,:));
    tof_front_status = squeeze(out.Sensor_ToF2.signals.values(:,4,:));
    tof_left  = squeeze(out.Sensor_ToF3.signals.values(:,1,:));
    tof_left_status = squeeze(out.Sensor_ToF3.signals.values(:,4,:));
    tof_right = squeeze(out.Sensor_ToF1.signals.values(:,1,:));
    tof_right_status = squeeze(out.Sensor_ToF1.signals.values(:,4,:));

    pos      = out.GT_position.signals.values;    % Nx3 ground truth
    rotQuat  = out.GT_rotation.signals.values;    % Nx4 (quaternions)
    timeVec  = out.GT_position.time;              % we’ll assume all the same length

    sensor_calibration = load("sensor_calibration.mat");

    %% Assume everything is the same length (no time alignment/truncation)
    N = length(pos);  % or size(pos,1)

    % If your data is truly all same length, no need to truncate
    % accel, gyro, mag, etc. must be length N as well

    GT = pos;  % Just store ground-truth positions for reference

    %% Frame Convention
    accel = (sensor_calibration.Rotw_a * accel')';
    gyro  = (sensor_calibration.Rotw_a * gyro')';
    mag   = (sensor_calibration.Rotw_mag * mag')';

    %% Sensor Calibration
    accel = accel - sensor_calibration.accel_bias;
    gyro  = gyro  - sensor_calibration.gyro_bias;
    mag   = (mag - sensor_calibration.b_mag) * sensor_calibration.A_mag;

    %% Convert GT quaternion to Euler angles
    eulAngles = quat2eul(rotQuat)  % [roll, pitch, yaw] in ZYX by default
    yawGT     = eulAngles(:,1);

    %% EKF Initialization (6D)
    % State = [ x, y, vx, vy, psi, dpsi ]
    X_Est = zeros(N, 6);
    P_Est = cell(N,1);

    % Use GT for initial position and yaw, zero velocity
    x0   = pos(1,1);
    y0   = pos(1,2);
    psi0 = yawGT(2);
    X_Est(1,:) = [ x0, y0, 0, 0, psi0, 0 ];

    % Give minimal uncertainty (adjust if needed)
    P_Est{1} = diag([0.01, 0.01, 0.05, 0.05, 0.01, 0.05]);

    % Process noise Q
    Q = diag([1e-6, 1e-6, 1e-3, 1e-3, 1e-5, 1e-3]);

    % Measurement noise R, scaled properly
    scaleFactors = [1, 0.1, 0.1, 0.1];
    Rdiag = [
        sensor_calibration.R_psi, ...
        sensor_calibration.R_tof_left, ...
        sensor_calibration.R_tof_middle, ...
        sensor_calibration.R_tof_right
    ];
    R = diag(scaleFactors .* Rdiag);

    %% Main EKF Loop
    for k = 1:N-1
        % We'll assume the dataset uses consistent time steps
        dt = timeVec(k+1) - timeVec(k);
        if dt <= 0, dt = 1e-2; end

        xk = X_Est(k,:)';
        [x_pred, F] = predictionStepSymbolic(xk, accel(k,:), gyro(k,:), dt);

        % Predicted covariance
        P_pred = F * P_Est{k} * F' + Q;

        % Build measurement
        [z_meas, H] = measurementModelOptim(x_pred, mag(k,:), tof_front(k), tof_left(k), tof_right(k));

        % Update
        [x_upd, P_upd] = updateStep(x_pred, P_pred, z_meas, H, R);

        % Store
        X_Est(k+1,:) = x_upd';
        P_Est{k+1}   = P_upd;
    end
end

%% ==============================================================
function [x_pred, F] = predictionStepSymbolic(xk, accel, gyro, dt)
    % xk(1) = x, xk(2) = y
    % xk(3) = vx, xk(4) = vy
    % xk(5) = psi, xk(6) = dpsi
    x   = xk(1);
    y   = xk(2);
    vx  = xk(3);
    vy  = xk(4);
    psi = xk(5);
    dpsi= xk(6);

    % We'll assume accel(3) is forward acceleration in your chosen frame
    af  = accel(3);
    gyZ = gyro(3);

    c = cos(psi);
    s = sin(psi);

    x_pred = zeros(6,1);

    % x position
    x_pred(1) = x + vx*dt + 0.5*c*af*dt^2;
    % y position
    x_pred(2) = y + vy*dt + 0.5*s*af*dt^2;
    % vx
    x_pred(3) = vx + c*af*dt;
    % vy
    x_pred(4) = vy + s*af*dt;
    % psi
    x_pred(5) = psi + dpsi*dt;
    % dpsi
    x_pred(6) = gyZ;  % new yaw rate from the gyro

    % Jacobian (6x6)
    F = eye(6);

    % partial x wrt vx
    F(1,3) = dt;
    % partial x wrt psi
    F(1,5) = -0.5*s*af*dt^2;

    % partial y wrt vy
    F(2,4) = dt;
    F(2,5) =  0.5*c*af*dt^2;

    % partial vx wrt psi
    F(3,5) = -s*af*dt;

    % partial vy wrt psi
    F(4,5) =  c*af*dt;

    % partial psi wrt dpsi
    F(5,6) = dt;

    % dpsi wrt states is 0 in this model (using direct assignment from gyroZ)
end

%% ==============================================================
function [z_meas, H] = measurementModelOptim(x_pred, mag, d1, d2, d3)
    % We measure yaw from magnetometer plus 3 ToF distances
    psi_meas = atan2(mag(2), mag(1));

    % Build z_meas
    z_meas = [psi_meas; d1; d2; d3];

    % Predicted yaw = x_pred(5)
    psi = x_pred(5);
    x   = x_pred(1);
    y   = x_pred(2);

    offsets = [0, pi/2, -pi/2];
    angles  = psi + offsets;

    % We'll compute partial derivatives for ToF
    d_pred = zeros(3,1);
    H = zeros(4,6);  % 4 measurements, 6 states

    for i = 1:3
        theta = angles(i);
        [t, ~, dt_dx, dt_dy, dt_dpsi] = rayBoxIntersection(x, y, theta);
        d_pred(i) = t;
        row = i+1;  % rows 2..4 in z_meas
        H(row,1) = dt_dx;     % partial wrt x
        H(row,2) = dt_dy;     % partial wrt y
        H(row,5) = dt_dpsi;   % partial wrt psi
    end

    % row 1 = yaw partial
    % yaw_meas depends only on psi => partial(psi_meas)/partial(psi_est) = 1
    H(1,5) = 1;
end

%% ==============================================================
function [x_upd, P_upd] = updateStep(x_pred, P_pred, z_meas, H, R)
    % Predicted measurement
    psi_est = x_pred(5);
    x_est   = x_pred(1);
    y_est   = x_pred(2);

    offsets = [0, pi/2, -pi/2];
    angles  = psi_est + offsets;
    d_pred  = zeros(3,1);

    for i = 1:3
        d_pred(i) = rayBoxIntersection(x_est, y_est, angles(i));
    end

    z_pred = [psi_est; d_pred];

    % innovation
    y_tilde = z_meas - z_pred;

    % wrap yaw difference
    y_tilde(1) = wrapToPi(y_tilde(1));

    % Compute Kalman gain
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;

    % update state
    x_upd = x_pred + K * y_tilde;
    % wrap final yaw
    x_upd(5) = wrapToPi(x_upd(5));

    % update covariance
    I = eye(length(x_pred));
    P_upd = (I - K * H) * P_pred;
end

%% ==============================================================
function [t, branch, dt_dx, dt_dy, dt_dpsi] = rayBoxIntersection(x, y, theta)
    x_left   = -1.2; x_right = 1.2;
    y_bottom = -1.2; y_top   = 1.2;

    cth = cos(theta);
    sth = sin(theta);

    if cth > 0
        x_wall = x_right;
    elseif cth < 0
        x_wall = x_left;
    else
        x_wall = NaN;
    end

    if ~isnan(x_wall) && abs(cth) > 1e-6
        t_v = (x_wall - x) / cth;
    else
        t_v = inf;
    end

    if sth > 0
        y_wall = y_top;
    elseif sth < 0
        y_wall = y_bottom;
    else
        y_wall = NaN;
    end

    if ~isnan(y_wall) && abs(sth) > 1e-6
        t_h = (y_wall - y) / sth;
    else
        t_h = inf;
    end

    if t_v <= 0, t_v = inf; end
    if t_h <= 0, t_h = inf; end

    if t_v < t_h
        t     = t_v;
        branch= 1;
        dt_dx = -1/cth;
        dt_dy = 0;
        dt_dtheta = (x_wall - x) * sth / (cth^2);
    else
        t     = t_h;
        branch= 2;
        dt_dx = 0;
        dt_dy = -1/sth;
        dt_dtheta = - (y_wall - y) * cth / (sth^2);
    end
    dt_dpsi = dt_dtheta;
end
