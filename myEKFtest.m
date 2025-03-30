    function [X_Est, P_Est, GT] = myEKF(out, bias, T)
    
        % ------------------ Sensor Data Extraction ------------------ %
        accel = squeeze(permute(out.Sensor_ACCEL.signals.values, [3, 2, 1]));
        gyro  = squeeze(permute(out.Sensor_GYRO.signals.values, [3, 2, 1]));
        mag   = squeeze(permute(out.Sensor_MAG.signals.values, [3, 2, 1]));
        tof1  = squeeze(out.Sensor_ToF1.signals.values(:,1,:));
        tof2  = squeeze(out.Sensor_ToF2.signals.values(:,1,:));
        tof3  = squeeze(out.Sensor_ToF3.signals.values(:,1,:));
        pos   = out.GT_position.signals.values;
        rot   = out.GT_rotation.signals.values;
        timeVec = out.GT_position.time;
    
        % ------------------ Data Alignment ------------------ %
        N = min([size(accel,1), size(gyro,1), size(mag,1), size(tof1,1), size(timeVec,1)]);
        pos = pos(1:N,:);
        timeVec = timeVec(1:N);
    
        % ------------------ Sensor Calibration ------------------ %

        Rwb_accel = [0 1 0; 
                     0 0 1; 
                    -1 0 0];

        % Magnetometer

        mag_calibrated = (mag - bias') * T;

        % Scale to match Earth's field magnitude (~48µT in London)
        H_ref = 48e-6;

        mag_calibrated = H_ref * (mag_calibrated ./ vecnorm(mag_calibrated, 2, 2));

        mag_corrected = (Rwb_accel * mag_calibrated')';  % Rotate to world frame
    
        % ------------------ EKF Initialization ------------------ %
        X_Est = zeros(N, 8);  % [x y vx vy psi dpsi gyro_bias accel_bias_x accel_bias_y]
        P_Est = cell(N,1);
        GT = pos;
    
        X_Est(1,:) = [pos(1,1:2), 0, 0, 0, 0, 0, 0, 0];  % Initial state
        P_Est{1} = diag([0.01, 0.01, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001]);
    
        Q = diag([1e-6, 1e-6, 1e-3, 1e-3, 1e-6, 1e-3, 1e-8, 1e-8]);  % Process noise
        R = diag([(1*pi/180)^2, (0.1*pi/180)^2, 0.05^2, 0.05^2, 0.05^2]);  % Measurement noise
    
        % ------------------ EKF Loop ------------------ %
        for k = 1:N-1
            dt = timeVec(k+1) - timeVec(k);
            if dt <= 0, dt = 1e-2; end
    
            % Prediction
            xk = X_Est(k,:)';
            [x_pred, F] = predictionStep(xk, accel(k,:), gyro(k,:), dt);
    
            % Measurement Update
            [z_meas, H] = measurementModel(x_pred, mag_corrected(k,:), tof1(k), tof2(k), tof3(k));
            [x_upd, P_upd] = updateStep(x_pred, P_Est{k}, z_meas, H, R, Q, dt);
    
            X_Est(k+1,:) = x_upd';
            P_Est{k+1} = P_upd;
        end
    end
    
    function [x_pred, F] = predictionStep(xk, accel, gyro, dt)
        % Unpack state
        psi = xk(5);
        c = cos(psi); s = sin(psi);
        gyro_bias = xk(7);
        accel_bias = xk(8:9);
    
        % Compensate IMU biases
        gy = gyro(3) - gyro_bias;
        af = accel(3) - accel_bias(1);
        as = -accel(1) - accel_bias(2);
    
        % World-frame acceleration
        ax_w = c*af - s*as;
        ay_w = s*af + c*as;
    
        % State prediction
        x_pred = xk;
        x_pred(1) = xk(1) + xk(3)*dt + 0.5*ax_w*dt^2;  % x
        x_pred(2) = xk(2) + xk(4)*dt + 0.5*ay_w*dt^2;  % y
        x_pred(3) = xk(3) + ax_w*dt;  % vx
        x_pred(4) = xk(4) + ay_w*dt;  % vy
        x_pred(5) = xk(5) + xk(6)*dt;  % psi
        x_pred(6) = gy;  % dpsi
    
        % Jacobian
        F = eye(8);
        F(1,3) = dt;
        F(1,5) = 0.5*dt^2*(-s*af - c*as);
        F(2,4) = dt;
        F(2,5) = 0.5*dt^2*(c*af - s*as);
        F(5,6) = dt;
    end
    
    function [z_meas, H] = measurementModel(x_pred, mag, tof1, tof2, tof3)
        % Yaw from magnetometer
        mx = mag(1); mz = mag(3);
        psi_meas = atan2(mx, mz);
    
        % ToF measurements
        z_meas = [psi_meas; x_pred(6); tof1; tof2; tof3];
    
        % Jacobian
        H = zeros(5,8);
        H(1,5) = 1;  % d(psi_meas)/d(psi)
        H(2,6) = 1;  % d(dpsi)/d(dpsi)
        H(3:5,1:2) = [0 -1; 1 0; -1 0];  % ToF model
    end