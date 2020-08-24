%% Load the raw data
datadir = 'data_RGCs/';  
load([datadir, 'Stim']);    
load([datadir,'stimtimes']); 
load([datadir, 'SpTimes']); 
ncells = length(SpTimes);  % number of neurons 

dtStim = (stimtimes(2)-stimtimes(1));
nT = size(Stim,1);

%% Bin spike trains 
tbins = (.5:nT)*dtStim; 
sps = zeros(nT,ncells);
for jj = 1:ncells
    sps(:,jj) = hist(SpTimes{jj},tbins)';  % bin spike train
end

clf;
nlags = 30; % number of lags
for ii = 1:ncells
    for jj = ii:ncells
        % cross-correlation of neuron i with neuron j
        xc = xcorr(sps(:,ii),sps(:,jj),nlags,'unbiased');
        if ii==jj, xc(nlags+1) = 0;
        end
        
        subplot(ncells,ncells,(ii-1)*ncells+jj);
        plot((-nlags:nlags)*dtStim,xc,'.-','markersize',20); 
        axis tight; drawnow;
        title(sprintf('cells (%d,%d)',ii,jj)); axis tight;
        xlabel('time shift (s)');
    end
end
saveas(gcf,"1.jpg")

%% Build design matrix: single-neuron GLM with spike-history
cellnum = 2; 
% the number of time bins of stimulus for predicting spikes
ntfilt = 25; 
% the number of time bins of history spikes 
nthist = 20;

% stimulus densign matrix(use hankel matrix)
paddedStim = [zeros(ntfilt-1,1); Stim];
Xstim = hankel(paddedStim(1:end-ntfilt+1), Stim(end-ntfilt+1:end));

% spike-history design matrix
paddedSps = [zeros(nthist,1); sps(1:end-1,cellnum)];
Xsp = hankel(paddedSps(1:end-nthist+1), paddedSps(end-nthist+1:end));
% final design matrix
Xdsgn = [Xstim,Xsp];

subplot(111); 
imagesc(1:(ntfilt+nthist), 1:50, Xdsgn(1:50,:));
xlabel('regressor');
ylabel('time bin of response');
title('design matrix with spike history dependence');
% saveas(gcf,'2.jpg')
%% fit GLM with spike history dependence 
% raw GLM
pGLMwts0 = glmfit(Xstim,sps(:,cellnum),'poisson');
pGLMconst0 = pGLMwts0(1);
pGLMfilt0 = pGLMwts0(2:end);

% fit GLM with spike history dependence
pGLMwts1 = glmfit(Xdsgn,sps(:,cellnum),'poisson');
pGLMconst1 = pGLMwts1(1);
pGLMfilt1 = pGLMwts1(2:1+ntfilt);
pGLMhistfilt1 = pGLMwts1(ntfilt+2:end);

%% plots comparing filters
ttk = (-ntfilt+1:0)*dtStim; 
tth = (-nthist:-1)*dtStim; 

clf; subplot(211); 
h = plot(ttk,ttk*0,'k--',ttk,pGLMfilt0, 'o-',ttk,pGLMfilt1,'o-','linewidth',2);
legend(h(2:3), 'GLM', 'GLM with spike history dependence','location','northwest');axis tight;
title('stimulus filters'); ylabel('weight');
xlabel('time before spike (s)');

subplot(212); 
colr = get(h(3),'color');
h = plot(tth,tth*0,'k--',tth,pGLMhistfilt1, 'o-');
set(h(2), 'color', colr, 'linewidth', 2); 
title('spike history filter'); 
xlabel('time before spike (s)');
ylabel('weight'); axis tight;
% saveas(gcf,'3.jpg')

%% Plot predicted rate out of the two models
ratepred0 = exp(pGLMconst0 + Xstim*pGLMfilt0);
ratepred1 = exp(pGLMconst1 + Xdsgn*pGLMwts1(2:end));

iiplot = 1:120; ttplot = iiplot*dtStim;
subplot(111);
stem(ttplot,sps(iiplot,cellnum), 'k'); hold on;
plot(ttplot,ratepred0(iiplot),ttplot,ratepred1(iiplot), 'linewidth', 2);
hold off;  axis tight;
legend('spikes', 'GLM', 'GLM with spike history dependence');
xlabel('time (s)');
title('spikes and rate predictions');
ylabel('spikes / bin');
% saveas(gcf,'4.jpg')

%% fit coupled GLM for multiple-neuron responses 
Xspall = zeros(nT,nthist,ncells);
for jj = 1:ncells
    paddedSps = [zeros(nthist,1); sps(1:end-1,jj)];
    Xspall(:,:,jj) = hankel(paddedSps(1:end-nthist+1),paddedSps(end-nthist+1:end));
end

Xspall = reshape(Xspall,nT,[]);
Xdsgn2 = [Xstim, Xspall]; % this design matrix consists of stimulus and spike history of all neurons

clf; % Let's visualize 50 time bins of full design matrix
imagesc(1:1:(ntfilt+nthist*ncells), 1:50, Xdsgn2(1:50,:));
title('design matrix');
xlabel('regressor');
ylabel('time bin of response');
saveas(gcf,'2.jpg')

%% Fit the model (stim filter, sphist filter, coupling filters) for one neuron 
pGLMwts2 = glmfit(Xdsgn2,sps(:,cellnum),'poisson');
pGLMconst2 = pGLMwts2(1);
pGLMfilt2 = pGLMwts2(2:1+ntfilt);
pGLMhistfilts2 = pGLMwts2(ntfilt+2:end);
pGLMhistfilts2 = reshape(pGLMhistfilts2,nthist,ncells);
 
%% fitted filters and rate prediction

subplot(211); 
h = plot(ttk,ttk*0,'k--',ttk,pGLMfilt0, 'o-',ttk,pGLMfilt1,...
    ttk,pGLMfilt2,'o-','linewidth',2); axis tight; 
legend(h(2:4), 'GLM', 'GLM with spike history dependence','GLM with multi-spike history dependence', 'location','northwest');
title(['stimulus filter: cell ' num2str(cellnum)]); ylabel('weight'); 
xlabel('time before spike (s)');

subplot(212); 
colr = get(h(3),'color');
h = plot(tth,tth*0,'k--',tth,pGLMhistfilts2,'linewidth',2);
legend(h(2:end),'from 1', 'from 2', 'from 3', 'from 4', 'location', 'northwest');
title(['spike history filters: into cell ' num2str(cellnum)]); axis tight;
xlabel('time before spike (s)');
ylabel('weight');
saveas(gcf,'3.jpg')

% predicted spike rate on training data
ratepred2 = exp(pGLMconst2 + Xdsgn2*pGLMwts2(2:end));

iiplot = 1:60; ttplot = iiplot*dtStim;
subplot(111);
stem(ttplot,sps(iiplot,cellnum), 'k'); hold on;
plot(ttplot,ratepred0(iiplot),ttplot,ratepred1(iiplot),...
    ttplot,ratepred2(iiplot), 'linewidth', 2);
hold off;  axis tight;
legend('spikes', 'GLM', 'GLM with spike history dependence', 'GLM with multi-spike history dependence', 'location', 'northwest');
xlabel('time (s)');
title('spikes and rate predictions');
ylabel('spikes / bin');
saveas(gcf,'4.jpg')

%% 6. Model comparison: log-likelihoood and AIC

LL_stimGLM = sps(:,cellnum)'*log(ratepred0) - sum(ratepred0);
LL_histGLM = sps(:,cellnum)'*log(ratepred1) - sum(ratepred1);
LL_coupledGLM = sps(:,cellnum)'*log(ratepred2) - sum(ratepred2);

nsp = sum(sps(:,cellnum));
ratepred_const = nsp/nT;  % mean number of spikes / bin
LL0 = nsp*log(ratepred_const) - nT*sum(ratepred_const);

SSinfo_stimGLM = (LL_stimGLM - LL0)/nsp/log(2);
SSinfo_histGLM = (LL_histGLM - LL0)/nsp/log(2);
SSinfo_coupledGLM = (LL_coupledGLM - LL0)/nsp/log(2);

% Compute AIC
AIC0 = -2*LL_stimGLM + 2*(1+ntfilt); 
AIC1 = -2*LL_histGLM + 2*(1+ntfilt+nthist);
AIC2 = -2*LL_coupledGLM + 2*(1+ntfilt+ncells*nthist);
AICmin = min([AIC0,AIC1,AIC2]); % the minimum of these

fprintf('\n AIC comparison (smaller is better):\n ---------------------- \n');
fprintf('stim-GLM: %.1f\n',AIC0-AICmin);
fprintf('hist-GLM: %.1f\n',AIC1-AICmin);
fprintf('coupled-GLM: %.1f\n',AIC2-AICmin);
