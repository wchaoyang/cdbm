function preprocess_dataset(datapath, imgformat, outpath)
    dirlist = dir(sprintf('%s/*.%s', datapath, imgformat));
    numimgs = length(dirlist);
    dataset = cell(numimgs, 1);
    for i=1:numimgs
        clc
        fprintf('preprocessing img: %d\\%d', numimgs, i)
        img = imread(sprintf('%s/%s', datapath, dirlist(i).name));
        img = imwhiten(img);
        % some trick implememt in Lee's implementation.
        img = img - mean(mean(img));
        img = img/sqrt(mean(mean(img.^2)));
        dataset{i} = sqrt(0.1)*img;% just for some trick?
    end
    fprintf('\n')
    fprintf('saving ....\n')
    save(outpath, 'dataset')
end