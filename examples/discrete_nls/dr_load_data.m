clearvars; close all; clc; format long;

% Load the file:
load data_AL.mat;

% Note that:
%
% 1) m is the number of lattice points s.t:
%
% n = -(m-1)/2:1:(m-1)/2.
% 
% 2) tout and yout are the output vectors of 
% the integrator.
%

% Create the complex field and make some plots:
     nn = -(m-1)/2:1:(m-1)/2;
ufield = yout(:,1:m) + 1i * yout(:,m+1:2*m);

fig1= figure;
subplot(1,2,1);
imagesc(nn,tout,abs(ufield).^2); colorbar; colormap;
xlabel('$n$','interprete','latex');
ylabel('$t$','interpreter','latex');
title('$|\psi_{n}(t)|^2$','interpreter','latex');
set(gca,'ydir','normal'); % Reverse the y (time) - axis.
set(gca,'fontsize',20,'fontname','times');

subplot(1,2,2);
% Here, we plot the modulus squared of the
% wavefunction at n=0, i.e., at the center of
% the spatial (yet discrete) domain. We see
% the periodicity over time!
plot(tout,abs(ufield(:,51)).^2,'linewidth',2);
xlabel('$t$','interpreter','latex');
ylabel('$|\psi_{0}(t)|^{2}$','interpreter','latex');
set(gca,'fontsize',20,'fontname','times');

set(fig1,'position',[100 100 1500 500]);