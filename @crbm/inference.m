function [H, HP] = inference(obj, V)
    ws = sqrt(size(obj.W,1));
    numbases = size(obj.W,3);
    numchannel = size(obj.W,2);

    poshidexp2 = zeros(size(V,1)-ws+1, size(V,2)-ws+1, numbases);
    for c=1:numchannel
        F = reshape(obj.W(end:-1:1, c, :),[ws,ws,numbases]);
        poshidexp2 = poshidexp2 + conv2_mult(V(:,:,c), F, 'valid');
    end
    
    if obj.vtype == obj.GAUSSIAN 
        for b=1:numbases
            poshidexp2(:,:,b) = 1/(obj.sigma^2).*(poshidexp2(:,:,b) + obj.hbias(b));
        end
    elseif obj.vtype == obj.BINARY
        for b=1:numbases
            poshidexp2(:,:,b) = poshidexp2(:,:,b) + obj.hbias(b);
        end
    else
        error('visible unit type: %s not supported!', obj.vtype)
    end
    
    [H, HP] = prb_pool(poshidexp2, obj.spacing);
end

function [H, HP] = prb_pool(poshidexp, spacing)
    % poshidexp is 3d array
    poshidprobs = exp(poshidexp);
    poshidprobs_mult = zeros(spacing^2+1, size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,3)/spacing^2);
    poshidprobs_mult(end,:) = 1;
    % TODO: replace this with more realistic activation, bases..
    for c=1:spacing
        for r=1:spacing
            temp = poshidprobs(r:spacing:end, c:spacing:end, :);
            poshidprobs_mult((c-1)*spacing+r,:) = temp(:);
        end
    end

    % [S P] = multrand2(poshidprobs_mult');
    [S1 P1] = multrand2(poshidprobs_mult');
    S = S1';
    P = P1';
    clear S1 P1

    % convert back to original sized matrix
    H = zeros(size(poshidexp));
    HP = zeros(size(poshidexp));
    for c=1:spacing
        for r=1:spacing
            H(r:spacing:end, c:spacing:end, :) = reshape(S((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
            HP(r:spacing:end, c:spacing:end, :) = reshape(P((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
        end
    end
end