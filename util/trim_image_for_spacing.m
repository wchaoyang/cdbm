function img = trim_image_for_spacing(img, fsize, spacing)
    if length(fsize) == 1
        fh = fsize;
        fw = fsize;
    else
        fh = fsize(1);
        fw = fsize(2);
    end
    
    if mod(size(img,1)-fh+1, spacing)~=0
        n = mod(size(img,1)-fh+1, spacing);
        img(1:floor(n/2), :, :) = [];
        img(end-ceil(n/2)+1:end, :, :) = [];
    end
    
    if mod(size(img,2)-fw+1, spacing)~=0
        n = mod(size(img,2)-fw+1, spacing);
        img(:,1:floor(n/2), :) = [];
        img(:,end-ceil(n/2)+1:end, :) = [];
    end
end