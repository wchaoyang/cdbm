function obj = train(obj, trainset, pars, numepoch, outpath)
    if ~isfield(pars, 'sigma_start'), pars.sigma_start = 0.2; end
    if ~isfield(pars, 'sigma_stop'), pars.sigma_stop = 0.1; end
    if ~isfield(pars, 'K_CD'), pars.K_CD = 1; end
    if ~isfield(pars, 'bias_mode'), pars.bias_mode = 'simple'; end
    if ~isfield(pars, 'pbias'), pars.pbias = 0.002; end
    if ~isfield(pars, 'pbias_lb'), pars.pbias = 0.002; end
    if ~isfield(pars, 'pbias_lambda'), pars.pbias_lambda = 5; end
    if ~isfield(pars, 'epsilon'), pars.epsilon = 0.01; end
    if ~isfield(pars, 'l2reg'), pars.l2reg = 0.01; end
    if ~isfield(pars, 'num_trials'), pars.num_trials = numepoch; end
    if ~isfield(pars, 'batch_size'), pars.batch_size = 100; end
    if ~isfield(pars, 'patch_size'), pars.patch_size = 70; end
    if ~isfield(pars, 'initial_momentum'), pars.initial_momentum = 0.5; end
    if ~isfield(pars, 'final_momentum'), pars.final_momentum = 0.9; end
    if ~isfield(pars, 'fliplr'), pars.fliplr = true; end
    disp(pars)
    
    ws = sqrt(size(obj.W, 1));
    error_history = [];
    sparsity_history = [];
    
    Winc=0;
    vbiasinc=0;
    hbiasinc=0;

    for t=1:pars.num_trials
        % Take a random permutation of the samples
        tic;
        ferr_current_iter = [];
        sparsity_curr_iter = [];
        idx_batch = randsample(length(trainset), pars.batch_size, length(trainset)<pars.batch_size);
        for i = 1:length(idx_batch)
            %%%%%%%% PRE-PROCESSING DATA %%%%%%%%%%%%%%%%%%%%%%%%
            x2d = trainset{idx_batch(i)};
            rows = size(x2d,1);
            cols = size(x2d,2);
            if rows>pars.patch_size && cols>pars.patch_size
                    rowidx = ceil(rand*(rows-2*ws-pars.patch_size))+ws + (1:pars.patch_size);
                    colidx = ceil(rand*(cols-2*ws-pars.patch_size))+ws + (1:pars.patch_size);
                    x2d = x2d(rowidx, colidx);
            end
            % trim image for fix convolution and spacing
            x2d = trim_image_for_spacing_fixconv(x2d, ws, obj.spacing);
            % flip data horizontaly
            if rand()>0.5 && pars.fliplr,
                x2d = fliplr(x2d);
            end
            if obj.vtype == obj.GAUSSIAN
                x2d = x2d - mean(x2d(:));
            end
            % update rbm
            [ferr, dW, dh, dv, poshidprobs, ~]= fobj_crbm_CD_LB_sparse(obj, x2d, pars);
            ferr_current_iter = [ferr_current_iter, ferr];
            sparsity_curr_iter = [sparsity_curr_iter, mean(poshidprobs(:))];
            
            if t<5,
                momentum = pars.initial_momentum;
            else
                momentum = pars.final_momentum;
            end
            
            % update parameters
            Winc = momentum*Winc + pars.epsilon*dW;
            obj.W = obj.W + Winc;
            
            vbiasinc = momentum*vbiasinc + pars.epsilon*dv;
            obj.vbias = obj.vbias + vbiasinc;
            
            hbiasinc = momentum*hbiasinc + pars.epsilon*dh;
            obj.hbias = obj.hbias + hbiasinc;
            
            mean_err = mean(ferr_current_iter);
            mean_sparsity = mean(sparsity_curr_iter);

            if (obj.sigma > pars.sigma_stop) % stop decaying after some point
                obj.sigma = obj.sigma*0.99;
            end

            % figure(1), display_network(W);
            % figure(2), subplot(1,2,1), imagesc(imdata(rowidx, colidx)), colormap gray
            % subplot(1,2,2), imagesc(negdata), colormap gray
        end
        toc;

        error_history(t) = mean(ferr_current_iter);
        sparsity_history(t) = mean(sparsity_curr_iter);

        figure(1), obj.display_bases();
        % if mod(t,10)==0,
        %    saveas(gcf, sprintf('%s_%04d.png', fname_save, t));
        % end

        fprintf('epoch %d error = %g \tsparsity_hid = %g\n', t, mean(ferr_current_iter), mean(sparsity_curr_iter));
%         save(fname_mat, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');
%         disp(sprintf('results saved as %s\n', fname_mat));

%         if mod(t, 10) ==0
%             fname_timestamp_save = sprintf('%s_%04d.mat', fname_prefix, t);
%             save(fname_timestamp_save, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');
%         end
    end    
end





function [ferr dW_total dh_total dv_total poshidprobs poshidstates negdata] = ...
    fobj_crbm_CD_LB_sparse(obj, V, pars)

ws = sqrt(size(obj.W,1));

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do convolution/ get poshidprobs
[poshidstates poshidprobs] = obj.inference(V);
posprods = crbm_vishidprod_fixconv(V, poshidprobs, ws);
poshidact = squeeze(sum(sum(poshidprobs,1),2));

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
neghidstates = poshidstates;
for j=1:pars.K_CD  %% pars.K_CD-step contrastive divergence
    negdata = obj.reconstruct(neghidstates);
    % neghidprobs = crbm_inference(negdata, W, hbias_vec, pars);
    % neghidstates = neghidprobs > rand(size(neghidprobs));
    [neghidstates neghidprobs] = obj.inference(negdata);
    
end
negprods = crbm_vishidprod_fixconv(negdata, neghidprobs, ws);
neghidact = squeeze(sum(sum(neghidprobs,1),2));

ferr = mean( (V(:)-negdata(:)).^2 );

if 0
    figure(1), display_images(imdata)
    figure(2), display_images(negdata)

    figure(3), display_images(W)
    figure(4), display_images(posprods)
    figure(5), display_images(negprods)

    figure(6), display_images(poshidstates)
    figure(7), display_images(neghidstates)
end


%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if strcmp(pars.bias_mode, 'none')
    dhbias = 0;
    dvbias = 0;
    dW = 0;
elseif strcmp(pars.bias_mode, 'simple')
    dhbias = squeeze(mean(mean(poshidprobs,1),2)) - pars.pbias;
    dvbias = 0;
    dW = 0;
elseif strcmp(pars.bias_mode, 'hgrad')
    error('hgrad not yet implemented!');
elseif strcmp(pars.bias_mode, 'Whgrad')
    error('Whgrad not yet implemented!');
else
    error('wrong adjust_bias mode!');
end

numcases1 = size(poshidprobs,1)*size(poshidprobs,2);
% dW_total = (posprods-negprods)/numcases - l2reg*W - weightcost_l1*sign(W) - pars.pbias_lambda*dW;
dW_total1 = (posprods-negprods)/numcases1;
dW_total2 = - pars.l2reg*obj.W;
dW_total3 = - pars.pbias_lambda*dW;
dW_total = dW_total1 + dW_total2 + dW_total3;

dh_total = (poshidact-neghidact)/numcases1 - pars.pbias_lambda*dhbias;

dv_total = 0; %dv_total';

end




function vishidprod2 = crbm_vishidprod_fixconv(imdata, H, ws)

    numchannels = size(imdata,3);
    numbases = size(H,3);

    % tic
    % TODO: single channel version is not implemented yet.. Might need to
    % modify mexglx file
    selidx1 = size(H,1):-1:1;
    selidx2 = size(H,2):-1:1;
    vishidprod2 = zeros(ws,ws,numchannels,numbases);

    if numchannels==1
        vishidprod2 = conv2_mult(imdata, H(selidx1, selidx2, :), 'valid');
    else
        for b=1:numbases
            vishidprod2(:,:,:,b) = conv2_mult(imdata, H(selidx1, selidx2, b), 'valid');
        end
    end

    vishidprod2 = reshape(vishidprod2, [ws^2, numchannels, numbases]);

end
