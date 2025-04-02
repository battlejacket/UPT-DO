function plotDataset3D(plotVar, field)
    arguments
        plotVar
        field = nan
    end

    X=plotVar(:,1);
    Y=plotVar(:,2);
%    Z=plotVar(:,3);
%    scatter3(X,Y,Z,5,field)
    scatter(X,Y,5,field)
    
%     else
%         if isa(field, 'cell')
%             scatter(X,Y,5,field{index})
%         else
%             scatter(X,Y,5)
%         end
%     end
end