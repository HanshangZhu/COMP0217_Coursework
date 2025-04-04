% jacobian_prediction.m
% Auto-generated analytical Jacobian function for EKF prediction model
% Inputs:
%   x, y, vx, vy, psi, dpsi, gb, ab, dt, af
% Outputs:
%   F - 8x8 Jacobian matrix of the prediction model

function F = jacobian_prediction(x, y, vx, vy, psi, dpsi, gb, ab, dt, af)
    c = cos(psi);
    s = sin(psi);

    F = eye(8);

    % Partial derivatives for x position
    F(1,3) = dt;
    F(1,5) = -0.5 * s * af * dt^2;
    F(1,8) = -0.5 * c * dt^2;

    % Partial derivatives for y position
    F(2,4) = dt;
    F(2,5) =  0.5 * c * af * dt^2;
    F(2,8) = -0.5 * s * dt^2;

    % Partial derivatives for vx
    F(3,5) = -s * af * dt;
    F(3,8) = -c * dt;

    % Partial derivatives for vy
    F(4,5) =  c * af * dt;
    F(4,8) = -s * dt;

    % Partial for psi
    F(5,6) = dt;

    % Partial for dpsi (gy = gyro(3) - gb)
    F(6,7) = -1;
end