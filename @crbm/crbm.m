classdef crbm
    properties
        W
        vbias
        hbias
        vtype % deal with this latter
        sigma
        spacing
        
    end
    properties(Constant)
        GAUSSIAN = 0;
        BINARY = 1;
    end
    
    methods
        function obj = crbm(ws, numchannels, numbases, vtype)
            obj.W = 0.01*randn(ws^2, numchannels, numbases);
            obj.vbias = zeros(numchannels,1);
            obj.hbias = -0.1*ones(numbases,1);
            obj.sigma = 0.1;
            obj.spacing = 2;
            if numargs == 4
                if strcmpi(vtype, 'g')
                    obj.vtype = obj.GAUSSIAN;
                elseif strcmpi(vtype, 'b')
                    obj.vtype = obj.BINARY;
                else
                    error('visible unit type: %s not supported!', vtype)
                end
            end
        end
        
        display_bases(obj)
        
        obj = train(obj, trainset, pars, numepoch, outpath)
        
        [H, HP] = inference(obj, H)
        
        H = reconstruct(obj, V)
        
    end
end