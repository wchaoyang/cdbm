function V = reconstruct(obj, H)
    ws = sqrt(size(obj.W,1));
    patch_M = size(H,1);
    patch_N = size(H,2);
    numchannels = size(obj.W,2);
    numbases = size(obj.W,3);

    V = zeros(patch_M+ws-1, patch_N+ws-1, numchannels);

    for b = 1:numbases,
        F = reshape(obj.W(:,:,b),[ws,ws,numchannels]);
        V = V + conv2_mult(H(:,:,b), F, 'full');
    end
end