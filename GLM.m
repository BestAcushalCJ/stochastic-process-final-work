%% Load the raw data
datadir = 'data_RGCs/'; 
load([datadir, 'Stim']);    % stimulus (temporal binary white noise)
load([datadir,'stimtimes']); % stim frame times in seconds 
load([datadir, 'SpTimes']); % spike times (in units of stim frames)

% choose a cell, a cell coresponds to a cellnum
% 1-2 are OFF cells; 3-4 are ON cells.
cellnum = 4; 
tsp = SpTimes{cellnum}; % spike times

dtStim = (stimtimes(2)-stimtimes(1)); % time bin size for stimulus
RefreshRate = 1/dtStim; 
nT = size(Stim,1); % number of time bins in stimulus
nsp = length(tsp); % number of spikes

fprintf('--------------------------\n');
fprintf('Load RGC data: cell %d\n', cellnum);
fprintf('Number of stim frames: %d  (%.1f minutes)\n', nT, nT*dtStim/60);
fprintf('Time bin size: %.1f ms\n', dtStim*1000);
fprintf('Number of spikes: %d (mean rate=%.1f Hz)\n\n', nsp, nsp/nT*RefreshRate); 

% visualize sampled raw data
subplot(211);
iiplot = 1:120; % bins of sampled raw stimulus
ttplot = iiplot*dtStim; 
plot(ttplot,Stim(iiplot), 'linewidth', 2);  axis tight;
title('raw stimulus');
ylabel('stimulus intensity');
subplot(212);
tspplot = tsp((tsp>=ttplot(1))&(tsp<ttplot(end))); % extract the corresponding spikes
plot(tspplot, 1, 'ko', 'markerfacecolor', 'k');
set(gca,'xlim', ttplot([1 end])); % set the range of x axis
title('spike times'); xlabel('time (s)');
saveas(gcf,'1.jpg');

%% Bin the spike train 
tbins = (.5:nT)*dtStim; % time bin centers for spike train
sps = hist(tsp,tbins)';  % binned spike train

% plot sampled interval
subplot(111);
stem(ttplot,sps(iiplot), 'k', 'linewidth', 2);
title('spike counts');
ylabel('spike count'); xlabel('time (s)');
set(gca,'xlim', ttplot([1 end]), 'ylim', [0 3.5]);
saveas(gcf,'2.jpg');

%% Build the design matrix
% the number of time bins of stimulus to use for predicting spikes
ntfilt = 25; 

paddedStim = [zeros(ntfilt-1,1); Stim]; % pad early bins of stimulus with zero
Xdsgn = zeros(nT,ntfilt);
for j = 1:nT
    Xdsgn(j,:) = paddedStim(j:j+ntfilt-1)'; % grab last 'ntfilt' bins of stmiulus 
end

clf; imagesc(-ntfilt+1:0, 1:50, Xdsgn(1:50,:));
xlabel('lags before spike time bin');
ylabel('time bin of response'); % the response is spike count per bin
title('Design matrix');
saveas(gcf,'3.jpg');

%% Compute and visualize the spike-triggered average (STA) 
% When the stimulus is Gaussian white noise, the STA provides an unbiased
% estimator for the filter in a GLM / LNP model.
% ref:Chichilnisky, E. J . A simple white noise analysis of neuronal light 
% responses. Network Computation in Neural Systems, 2001, 12(2):199-213.
sta = (Xdsgn'*sps)/nsp;

ttk = (-ntfilt+1:0)*dtStim; 
plot(ttk,ttk*0, 'k--', ttk, sta, 'o-', 'linewidth', 2); axis tight;
title('STA'); xlabel('time before spike (s)');
saveas(gcf,'4.jpg');

%% whitened STA (fit to filter for least squares regression)
% If the stimuli are non-white, then the STA is generally a biased
% estimator for the linear filter. In this case we may wish to compute the
% "whitened" STA, which is also the maximum-likelihood estimator for the 
% filter of a GLM with "identity" nonlinearity and Gaussian noise 
% (also known as least-squares regression). 
% ref:Durbin J, Watson G S. Testing for serial correlation in least squares
% regression: I. Biometrika, 1950, 37(3/4): 409-428.

wsta = (Xdsgn'*Xdsgn)\sta*nsp;

h = plot(ttk,ttk*0, 'k--', ttk, sta./norm(sta), 'o-',...
    ttk, wsta./norm(wsta), 'o-', 'linewidth', 2); axis tight;
legend(h(2:3), 'STA', 'wSTA', 'location', 'northwest');
title('STA and whitened STA'); xlabel('time before spike (s)');
saveas(gcf,'5.jpg');

%% Predicting spikes with a least squares regression
% The whitened STA can actually be used to predict spikes because it
% corresponds to a proper estimate of the model parameters (i.e., for a
% Gaussian GLM). Let's inspect this prediction
% ref:Zuur A F, Ieno E N, Walker N J, et al. GLM and GAM for count data.
% Mixed effects models and extensions in ecology with R. Springer, New York, NY, 2009: 209-243.

sppred_lgGLM = Xdsgn*wsta;  % predicted spikes from least squares regression

% true spike train and prediction
stem(ttplot,sps(iiplot)); hold on;
plot(ttplot,sppred_lgGLM(iiplot),'linewidth',2); hold off;
title('linear least-squares regression: spike count prediction');
ylabel('spike count'); xlabel('time (s)');
set(gca,'xlim', ttplot([1 end]));
legend('spike count', 'linear LS regression');
saveas(gcf,'6.jpg');

%% Poisson GLM 
pGLMwts = glmfit(Xdsgn,sps,'poisson', 'constant', 'on');
pGLMconst = pGLMwts(1); 
pGLMfilt = pGLMwts(2:end); 
% predicted spike rate on training data
ratepred_pGLM = exp(pGLMconst + Xdsgn*pGLMfilt);

%% spike rate predictions
subplot(211);
h = plot(ttk,ttk*0, 'k--', ttk, wsta2./norm(wsta2), 'o-',...
    ttk, pGLMfilt./norm(pGLMfilt), 'o-', 'linewidth', 2); axis tight;
legend(h(2:3), 'linear LS regression', 'exp-poisson GLM', 'location', 'southwest');
title('linear LS regression and Poisson GLM filter estimates'); 
xlabel('time before spike (s)');

subplot(212);
stem(ttplot,sps(iiplot)); hold on;
h = plot(ttplot,sppred_lgGLM2(iiplot),ttplot,ratepred_pGLM(iiplot)); 
set(h, 'linewidth', 2);  hold off;
title('spike rate predictions');
ylabel('spikes / bin'); xlabel('time (s)');
set(gca,'xlim', ttplot([1 end]));
legend('spike count', 'linear LS regression', 'exp-poisson GLM');
saveas(gcf,'7.jpg');

%% Non-parametric estimate of the nonlinearity
% number of bins for parametrizing the nonlinearity f.  
nfbins = 25; 

rawfilteroutput = pGLMconst + Xdsgn*pGLMfilt;

% bin filter output and get bin index for each filtered stimulus
[cts,binedges,binID] = histcounts(rawfilteroutput,nfbins); 
fx = binedges(1:end-1)+diff(binedges(1:2))/2; 

% mean spike count in each bin
fy = zeros(nfbins,1); 
for jj = 1:nfbins
    fy(jj) = mean(sps(binID==jj));
end
fy = fy/dtStim; 
fnlin = @(x)(interp1(fx,fy,x,'nearest','extrap'));

subplot(111);
xx = binedges(1):.01:binedges(end);
plot(xx,exp(xx)/dtStim,xx,fnlin(xx),'linewidth', 2);
xlabel('filter output');
ylabel('rate (sp/s)');
legend('exponential f', 'nonparametric f', 'location', 'northwest');
title('comparison between nonlinearity');
saveas(gcf,'8.jpg')

%% 7. Quantifying performance: log-likelihood
ratepred_pGLM = exp(pGLMconst + Xdsgn*pGLMfilt); % rate under exp nonlinearity
LL_expGLM = sps'*log(ratepred_pGLM) - sum(ratepred_pGLM);

ratepred_pGLMnp = dtStim*fnlin(pGLMconst + Xdsgn*pGLMfilt); % rate under nonpar nonlinearity
LL_npGLM = sps(sps>0)'*log(ratepred_pGLMnp(sps>0)) - sum(ratepred_pGLMnp);

ratepred_const = nsp/nT;  % mean number of spikes / bin
LL0 = nsp*log(ratepred_const) - nT*ratepred_const;

SSinfo_expGLM = (LL_expGLM - LL0)/nsp/log(2);
SSinfo_npGLM = (LL_npGLM - LL0)/nsp/log(2);

subplot(111);
stem(ttplot,sps(iiplot)); hold on;
plot(ttplot,ratepred_pGLM(iiplot),ttplot,ratepred_pGLMnp(iiplot),'linewidth',2); 
hold off; title('spike rate predictions');
ylabel('spikes / bin'); xlabel('time (s)');
set(gca,'xlim', ttplot([1 end]));
legend('spike count', 'exp-GLM', 'np-GLM');
saveas(gcf,'9.jpg')

%% 8. Quantifying performance: AIC
% AIC = - 2*log-likelihood + 2 * number-of-parameters
AIC_expGLM = -2*LL_expGLM + 2*(1+ntfilt); 
AIC_npGLM = -2*LL_npGLM + 2*(1+ntfilt+nfbins);

fprintf('\n AIC comparison:\n ---------------------- \n');
fprintf('exp-GLM: %.1f\n',AIC_expGLM);
fprintf(' np-GLM: %.1f\n',AIC_npGLM);

%% 9. Simulating the GLM / making a raster plot
iiplot = 1:120; % time bins of stimulus to use
ttplot = iiplot*dtStim; 
StimRpt = Stim(iiplot);
nrpts = 50;  % number of repeats
frate = exp(pGLMconst+Xdsgn(iiplot,:)*pGLMfilt);% firing rate in each bin

subplot(611);
plot(ttplot,Stim(iiplot), 'linewidth', 2);  axis tight;
title('raw stimulus');
ylabel('stim intensity'); set(gca,'xticklabel', {});
subplot(612);
tspplot = tsp((tsp>=ttplot(1))&(tsp<ttplot(end)));
plot(tspplot, 1, 'ko', 'markerfacecolor', 'k');
set(gca,'xlim', ttplot([1 end]));
title('real spike times');
set(gca,'xticklabel', {});

spcounts = poissrnd(repmat(frate',nrpts,1));
subplot(6,1,3:6);
imagesc(ttplot,1:nrpts, spcounts);
ylabel('repeat index');
xlabel('time (s)');
title('exp-GLM spike trains');
saveas(gcf,'10.jpg')
